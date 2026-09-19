"""
Phase 1 unit tests — no GPU required.
Run from project root: python tests/test_phase1.py
"""

import random
import sys

sys.path.insert(0, ".")

print("=" * 55)
print("  LinguaFranca Phase 1 — Unit Tests")
print("=" * 55)

# ── imports ────────────────────────────────────────────────────
from utils.matching import (
    EntityMatcher,
    extract_noun_phrase_candidates,
    normalize,
)
from phase1_dataset.generate_cot import parse_hop_spans, parse_predicted_answer
from phase1_dataset.label_hops import label_example
from phase1_dataset.counterfactuals import find_entry_entity
from phase1_dataset.build_dataset import check_no_leakage, split_by_id

# ── Test 1: normalize() ────────────────────────────────────────
assert normalize("Christopher Nolan") == "christopher nolan"
assert normalize("New-York!") == "new york"
assert normalize("  multiple   spaces  ") == "multiple spaces"
print("[PASS] normalize()")

# ── Test 2: NP extraction ──────────────────────────────────────
nps = extract_noun_phrase_candidates(
    "The director of Inception is Christopher Nolan."
)
assert "Inception" in nps
assert "Christopher Nolan" in nps
print("[PASS] extract_noun_phrase_candidates()")

# ── Test 3: Tier-0 substring match ────────────────────────────
bare_matcher = EntityMatcher(aliases={})
r = bare_matcher.match(
    "Christopher Nolan",
    "Inception was directed by Christopher Nolan.",
)
assert r.matched and r.method == "normalized_string", f"Got {r}"
print("[PASS] Tier-0 substring match")

# ── Test 4: Tier-0 miss ────────────────────────────────────────
r = bare_matcher.match(
    "Christopher Nolan",
    "Inception was directed by James Cameron.",
)
assert not r.matched, f"Should fail: {r}"
print("[PASS] Tier-0 correctly misses wrong entity")

# ── Test 5: Tier-1 alias match ────────────────────────────────
alias_matcher = EntityMatcher(
    aliases={"Christopher Nolan": ["Chris Nolan", "C. Nolan", "Nolan"]}
)
r = alias_matcher.match(
    "Christopher Nolan",
    "The film was directed by Chris Nolan.",
)
assert r.matched and r.method == "wikidata_alias", f"Got {r}"
print("[PASS] Tier-1 alias match")

# ── Test 6: hop XML parser ────────────────────────────────────
gen = (
    "<hop1>Inception was directed by Christopher Nolan.</hop1>"
    "<hop2>His mother is Lynda Nolan.</hop2>"
    "<answer>Lynda Nolan</answer>"
)
hops = parse_hop_spans(gen)
assert len(hops) == 2
assert hops[0]["hop_idx"] == 1
assert "Christopher Nolan" in hops[0]["text"]
assert hops[1]["hop_idx"] == 2
ans = parse_predicted_answer(gen)
assert ans == "Lynda Nolan", f"Got {ans}"
print("[PASS] parse_hop_spans() and parse_predicted_answer()")

# ── Test 7: label_example — all hops succeed ─────────────────
BASE_EX = {
    "id": "test_001",
    "question": "Who is the mother of the director of Inception?",
    "gold_answer": "Lynda Nolan",
    "reasoning_graph": [
        {"hop": 1, "gold_entity": "Christopher Nolan"},
        {"hop": 2, "gold_entity": "Lynda Nolan"},
    ],
    "generated_cot": (
        "<hop1>Inception was directed by Christopher Nolan.</hop1>"
        "<hop2>His mother is Lynda Nolan.</hop2>"
    ),
    "hop_spans": [
        {"hop_idx": 1, "text": "Inception was directed by Christopher Nolan.",
         "char_start": 6, "char_end": 49},
        {"hop_idx": 2, "text": "His mother is Lynda Nolan.",
         "char_start": 62, "char_end": 87},
    ],
    "predicted_answer": "Lynda Nolan",
}
labeled = label_example(BASE_EX, alias_matcher)
h0, h1 = labeled["hops"][0], labeled["hops"][1]
assert h0["label"] == 0, f"hop1 should succeed: {h0}"
assert h1["label"] == 0, f"hop2 should succeed: {h1}"
assert labeled["first_fail_hop"] is None
assert labeled["final_answer_correct"] is True
print("[PASS] label_example() — all hops succeed")

# ── Test 8: label_example — failure at hop 1 ─────────────────
FAIL_EX = {
    **BASE_EX,
    "id": "test_002",
    "generated_cot": (
        "<hop1>Avatar was directed by James Cameron.</hop1>"
        "<hop2>His mother is Shirley Cameron.</hop2>"
    ),
    "hop_spans": [
        {"hop_idx": 1, "text": "Avatar was directed by James Cameron.",
         "char_start": 6, "char_end": 42},
        {"hop_idx": 2, "text": "His mother is Shirley Cameron.",
         "char_start": 55, "char_end": 84},
    ],
    "predicted_answer": "Shirley Cameron",
}
labeled_fail = label_example(FAIL_EX, alias_matcher)
assert labeled_fail["hops"][0]["label"] == 1, "hop1 should FAIL"
assert labeled_fail["first_fail_hop"] == 1
assert labeled_fail["final_answer_correct"] is False
print("[PASS] label_example() — failure detected at hop 1")

# ── Test 9: missing hop tag → treated as failure ──────────────
MISS_EX = {
    **BASE_EX,
    "id": "test_003",
    "generated_cot": (
        "The director of Inception is Christopher Nolan. "
        "His mother is Lynda Nolan."
    ),
    "hop_spans": [],
    "predicted_answer": None,
}
labeled_miss = label_example(MISS_EX, alias_matcher)
assert all(h["label"] == 1 for h in labeled_miss["hops"]), \
    "Missing hops should all be failures"
assert labeled_miss["first_fail_hop"] == 1
print("[PASS] Missing hop tags treated as failure")

# ── Test 10: split_by_id — no leakage, CF stays with base ────
fake_exs = (
    [{"id": f"q{i}", "first_fail_hop": None} for i in range(80)]
    + [{"id": f"q{i}_cf", "first_fail_hop": 1} for i in range(40)]
)
rng = random.Random(42)
tr, va, te = split_by_id(fake_exs, 0.6, 0.2, rng)
check_no_leakage(tr, va, te)

tr_ids  = {e["id"] for e in tr}
val_ids = {e["id"] for e in va}
te_ids  = {e["id"] for e in te}
for i in range(40):
    base = f"q{i}"
    cf   = f"q{i}_cf"
    if base in tr_ids:
        assert cf in tr_ids,  f"{cf} leaked out of train"
    if base in val_ids:
        assert cf in val_ids, f"{cf} leaked out of val"
    if base in te_ids:
        assert cf in te_ids,  f"{cf} leaked out of test"
print("[PASS] split_by_id() — no leakage, CF stays with base")

# ── Test 11: find_entry_entity ────────────────────────────────
graph = [
    {"hop": 1, "gold_entity": "Christopher Nolan"},
    {"hop": 2, "gold_entity": "Lynda Nolan"},
]
entry = find_entry_entity(
    "Who is the mother of the director of Inception?", graph
)
assert entry == "Inception", f"Got: {entry}"
print("[PASS] find_entry_entity()")

# ── Test 12: class balance checker ───────────────────────────
from phase1_dataset.build_dataset import check_class_balance
import io, logging

# Just verify it runs and logs without crashing
check_class_balance("test_split", [
    {"first_fail_hop": None},
    {"first_fail_hop": 1},
    {"first_fail_hop": None},
    {"first_fail_hop": 2},
])
print("[PASS] check_class_balance()")

print()
print("=" * 55)
print("  ALL 12 TESTS PASSED")
print("=" * 55)
