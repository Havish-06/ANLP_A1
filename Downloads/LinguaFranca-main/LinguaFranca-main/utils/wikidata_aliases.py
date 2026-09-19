"""
utils/wikidata_aliases.py
─────────────────────────────
Build and load a local alias table:  entity_name → [alias1, alias2, ...]

Strategy
────────
1. Check for a cached JSON file (fast path — no network call).
2. If cache is missing or specific entities are absent, query the Wikidata
   SPARQL endpoint (rate-limited to be polite).
3. Save/merge back to the cache.

The resulting dict is passed to EntityMatcher in matching.py.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "LinguaFranca-research/1.0 (contact: research@example.com)",
}

# SPARQL template: finds all English-language aliases for an entity whose
# English rdfs:label exactly matches the input string.
_SPARQL_TEMPLATE = """\
SELECT ?alias WHERE {{
  ?item rdfs:label "{entity}"@en .
  ?item skos:altLabel ?alias .
  FILTER(LANG(?alias) = "en")
}}
LIMIT 30
"""


# ─────────────────────────────────────────────────────────────────────────────
# Single-entity query
# ─────────────────────────────────────────────────────────────────────────────

def _query_wikidata(entity_name: str, timeout: int = 12) -> list[str]:
    """
    Query Wikidata for all English aliases of `entity_name`.
    Returns an empty list on any error.
    """
    query = _SPARQL_TEMPLATE.format(entity=entity_name.replace('"', '\\"'))
    try:
        resp = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        bindings = resp.json()["results"]["bindings"]
        return [b["alias"]["value"] for b in bindings]
    except Exception as exc:
        logger.debug("Wikidata query failed for %r: %s", entity_name, exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_alias_table(cache_path: str) -> dict[str, list[str]]:
    """
    Load the alias table from a local JSON cache.
    Returns an empty dict if the file does not exist.
    """
    path = Path(cache_path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_alias_table(
    entity_names: list[str],
    cache_path: str,
    rate_limit_sec: float = 0.5,
    force_refresh: bool = False,
) -> dict[str, list[str]]:
    """
    Ensure every entity in `entity_names` has an alias list in the cache.

    Parameters
    ----------
    entity_names : list[str]
        All canonical bridging entities found in the dataset.
    cache_path : str
        Path to local JSON cache file (created if absent).
    rate_limit_sec : float
        Seconds to sleep between SPARQL requests.
    force_refresh : bool
        If True, re-query all entities even if already cached.

    Returns
    -------
    dict[str, list[str]]
        Full alias table (existing + newly queried).
    """
    cache_file = Path(cache_path)
    existing: dict[str, list[str]] = {}

    if cache_file.exists() and not force_refresh:
        with open(cache_file, encoding="utf-8") as f:
            existing = json.load(f)

    to_query = [e for e in entity_names if e not in existing]

    if not to_query:
        logger.info("Alias cache is complete — no SPARQL queries needed.")
        return existing

    logger.info("Querying Wikidata for %d new entities…", len(to_query))
    for entity in tqdm(to_query, desc="Wikidata aliases"):
        existing[entity] = _query_wikidata(entity)
        time.sleep(rate_limit_sec)

    # Save merged table
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    logger.info("Alias table saved to %s  (%d entities)", cache_path, len(existing))
    return existing


def extract_all_gold_entities(labeled_examples: list[dict]) -> list[str]:
    """
    Collect every unique gold bridging entity across a list of labeled examples.
    Convenience helper so callers don't have to walk the schema themselves.
    """
    entities: set[str] = set()
    for ex in labeled_examples:
        for node in ex.get("reasoning_graph", []):
            ent = node.get("gold_entity", "")
            if ent:
                entities.add(ent)
    return sorted(entities)
