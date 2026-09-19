"""
utils/matching.py
─────────────────────
Three-tier (+ optional LLM-judge) entity PRESENCE CHECK for hop labeling.

Key design decision
───────────────────
This is NOT an extraction task.
Because 2WikiMultihopQA supplies the gold bridging entity from its reasoning
graph, we already know what the model *should* have said.  The question is
only: does the hop text mention or imply it?

Tier 0 — Normalized substring match         (fast, ~65 % of cases)
Tier 1 — Wikidata alias expansion            (catches "Chris Nolan" etc.)
Tier 2 — Sentence-BERT cosine on NP spans   (semantic paraphrase fallback)
Tier 3 — LLM-as-judge                       (budget-capped, truly ambiguous)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    matched: bool
    method: str          # "normalized" | "alias" | "sbert" | "llm" | "none"
    score: Optional[float] = None   # cosine score if method == "sbert"

    def __repr__(self) -> str:
        return f"MatchResult(matched={self.matched}, method={self.method!r}, score={self.score})"


# ─────────────────────────────────────────────────────────────────────────────
# Text normalization helpers
# ─────────────────────────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^\w\s]")
_WS    = re.compile(r"\s+")


def normalize(text: str) -> str:
    """
    Canonical form for substring matching:
      1. NFC unicode normalization
      2. Lowercase
      3. Strip punctuation
      4. Collapse whitespace
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def extract_noun_phrase_candidates(text: str) -> list[str]:
    """
    Lightweight regex-based NP extraction — avoids loading a full NER model.

    Extracts sequences of Title-Case words (a proxy for named entities) plus
    the full text as a catch-all.  These are the candidates compared against
    the gold entity embedding in tier-2 matching.

    Example:
        "The director of Inception is Christopher Nolan."
        → ["Inception", "Christopher Nolan", "The director of Inception is Christopher Nolan."]
    """
    # Consecutive Title-Case tokens  (e.g. "Christopher Nolan", "New York City")
    pattern = r"\b[A-Z][a-zA-Z'-]*(?:\s+[A-Z][a-zA-Z'-]*)*\b"
    spans = re.findall(pattern, text)
    # De-duplicate while preserving order
    seen: set[str] = set()
    unique = []
    for s in spans:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique + [text]   # full text as final fallback


# ─────────────────────────────────────────────────────────────────────────────
# LLM-judge type alias
# ─────────────────────────────────────────────────────────────────────────────

# Signature: (gold_entity: str, hop_text: str) -> bool
LLMJudge = Callable[[str, str], bool]


# ─────────────────────────────────────────────────────────────────────────────
# Main matcher
# ─────────────────────────────────────────────────────────────────────────────

class EntityMatcher:
    """
    Checks whether a *known* gold entity is present or implied in hop text.

    Parameters
    ----------
    aliases : dict[str, list[str]]
        Mapping of canonical entity name → list of known aliases.
        Built by utils/wikidata_aliases.py.
    sbert_model_name : str
        HuggingFace sentence-transformers model ID.
    sbert_threshold : float
        Cosine similarity threshold for tier-2 match.
    llm_judge : LLMJudge | None
        Optional callable for tier-3 LLM-as-judge.
    llm_budget : float
        Maximum fraction of examples routed to LLM judge.
    """

    def __init__(
        self,
        aliases: dict[str, list[str]],
        sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        sbert_threshold: float = 0.80,
        llm_judge: Optional[LLMJudge] = None,
        llm_budget: float = 0.05,
    ) -> None:
        self.aliases         = aliases
        self.sbert_threshold = sbert_threshold
        self.llm_judge       = llm_judge
        self.llm_budget      = llm_budget

        self._sbert_model_name = sbert_model_name
        self._sbert: Optional[SentenceTransformer] = None

        # Budget tracking
        self._total_calls = 0
        self._llm_calls   = 0

    # ── Lazy SBERT loader ───────────────────────────────────────────────────
    def _get_sbert(self) -> SentenceTransformer:
        if self._sbert is None:
            self._sbert = SentenceTransformer(self._sbert_model_name)
        return self._sbert

    # ── Public API ──────────────────────────────────────────────────────────
    def match(self, gold_entity: str, hop_text: str) -> MatchResult:
        """
        Run tiered matching.  Returns on the first hit.

        Parameters
        ----------
        gold_entity : str
            The canonical bridging entity from the reasoning graph.
        hop_text : str
            The full text of a single <hopN>...</hopN> block.
        """
        self._total_calls += 1

        # ── Tier 0: Normalized substring ────────────────────────────────────
        if normalize(gold_entity) in normalize(hop_text):
            return MatchResult(matched=True, method="normalized_string")

        # ── Tier 1: Wikidata alias expansion ────────────────────────────────
        for alias in self.aliases.get(gold_entity, []):
            if normalize(alias) in normalize(hop_text):
                return MatchResult(matched=True, method="wikidata_alias")

        # ── Tier 2: Sentence-BERT cosine over NP candidates ─────────────────
        candidates = extract_noun_phrase_candidates(hop_text)
        if candidates:
            sbert = self._get_sbert()
            gold_emb  = sbert.encode(gold_entity, convert_to_numpy=True,
                                     show_progress_bar=False)
            cand_embs = sbert.encode(candidates,   convert_to_numpy=True,
                                     show_progress_bar=False)
            # Cosine similarity  (safe division)
            norms      = np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-9
            gold_norm  = np.linalg.norm(gold_emb) + 1e-9
            scores     = (cand_embs / norms) @ (gold_emb / gold_norm)
            best_score = float(scores.max())
            if best_score >= self.sbert_threshold:
                return MatchResult(matched=True, method="sbert", score=best_score)

        # ── Tier 3: LLM-as-judge (budget-capped) ────────────────────────────
        llm_fraction = self._llm_calls / max(self._total_calls, 1)
        if self.llm_judge is not None and llm_fraction < self.llm_budget:
            self._llm_calls += 1
            decision = self.llm_judge(gold_entity, hop_text)
            return MatchResult(matched=decision, method="llm")

        return MatchResult(matched=False, method="none")

    def stats(self) -> dict[str, int | float]:
        return {
            "total_calls": self._total_calls,
            "llm_calls":   self._llm_calls,
            "llm_fraction": round(self._llm_calls / max(self._total_calls, 1), 4),
        }
