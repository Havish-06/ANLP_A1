# LinguaFranca

> **"Do Internal Failures Predict External Ones?"**  
> Causal Hop Localization for Faithful, Efficient Retrieval-Augmented Multi-Hop QA

---

## Overview

This repository implements the full data and modelling pipeline for the project. We test whether a lightweight probe on an LLM's hidden states can detect and localize a failing reasoning hop **at the hop boundary** — before subsequent steps are generated — and whether that signal is **causally** responsible for the failure (via activation patching).

### Five Phases
| Phase | Description | Location |
|---|---|---|
| 1 | **Data construction & hop labeling** | `phase1_dataset/` |
| 2 | **Internal-state probing** | `src/probe/` *(coming)* |
| 3 | **Causal validation (activation patching)** | `src/causal/` *(coming)* |
| 4 | **Naive Restart RAG intervention** | `src/rag/` *(coming)* |
| 5 | **Evaluation vs Doctor-RAG baseline** | `src/eval/` *(coming)* |

---

## Setup

```bash
pip install -r requirements.txt
```

For Llama-3.2-3B-Instruct you need a Hugging Face token with gated-model access:
```bash
huggingface-cli login
```

---

## Phase 1 — Dataset Construction

### What Happened
In Phase 1, we built the dataset required to train a probe that detects reasoning failures in LLMs based on their hidden states.

### Why Did We Do This?
Large Language Models usually answer questions correctly, meaning natural failures are rare. To train a balanced probe, we need both successful reasoning paths and failed ones, along with the model's internal "brain activity" (hidden states) during both.

### How Was It Done?
1. **Generated Reasoning**: Fed multi-hop questions to the model to generate Chain-of-Thought reasoning.
2. **Recorded States**: Saved the model's internal hidden states while it generated its reasoning.
3. **Labeled Hops**: Programmatically graded each reasoning hop as Success or Failure based on whether it retrieved the correct bridging entity.
4. **Counterfactuals**: Artificially induced failures by swapping entities in the prompt, forcing the model to fail, which balances our dataset.
5. **Split Data**: Separated the data into train, val, and test splits securely.

### What Should We Do Next?
**Phase 2 — Internal-state probing**: We will train a classifier probe on `train.jsonl` using the hidden states saved in `data/hidden_states/` to predict hop-level failures before they fully manifest in text.

### Commands

```bash
# Full pipeline (download → CoT → label → counterfactuals → split)
python phase1_dataset/build_dataset.py --config configs/data_config.yaml

# Dry-run (skips model inference, validates schema only)
python phase1_dataset/build_dataset.py --config configs/data_config.yaml --dry-run

# Individual steps
python phase1_dataset/download.py       --config configs/data_config.yaml
python phase1_dataset/generate_cot.py   --config configs/data_config.yaml
python phase1_dataset/label_hops.py     --config configs/data_config.yaml
python phase1_dataset/counterfactuals.py --config configs/data_config.yaml
```

### Output schema (`data/processed/*.jsonl`)
```json
{
  "id": "2wiki_00421",
  "source": "2wikimultihopqa",
  "question": "...",
  "gold_answer": "...",
  "reasoning_graph": [{"hop": 1, "gold_entity": "..."}],
  "generated_cot": "<hop1>...</hop1><hop2>...</hop2>",
  "hops": [{
    "hop_idx": 1,
    "text": "...",
    "bridging_entity_gold": "...",
    "bridging_entity_pred": "...",
    "match_method": "normalized_string",
    "label": 0
  }],
  "first_fail_hop": null,
  "is_counterfactual": false,
  "final_answer_correct": true
}
```

---

## Project Structure

> **Note:** The `data/` folder is heavily populated during Phase 1. Because of its large size (3+ GB), it is not tracked in this repository locally. The generated data is hosted securely on Kaggle at the [Phase 1 Dataset Link](https://www.kaggle.com/datasets/havishbalaga/linguafranca-phase1-data).

```
LinguaFranca/
├── configs/
│   └── data_config.yaml
├── data/                     ← [Phase 1 Kaggle Dataset](https://www.kaggle.com/datasets/havishbalaga/linguafranca-phase1-data)
│   ├── raw/                  ← (Generated when running Phase 1)
│   ├── processed/            ← (Generated when running Phase 1)
│   └── hidden_states/        ← (Generated when running Phase 1)
├── src/
│   ├── utils/
│   │   ├── matching.py       
│   │   └── wikidata_aliases.py
│   └── data/
│       ├── download.py
│       ├── generate_cot.py
│       ├── label_hops.py
│       ├── counterfactuals.py
│       └── build_dataset.py
├── tests/                    ← Unit tests
├── kaggle_phase1.ipynb       ← Kaggle notebook for running Phase 1
├── kaggle_run.py             ← Script for Kaggle execution
├── LinguaFranca-Proposal.pdf ← Project Proposal
└── requirements.txt
```
