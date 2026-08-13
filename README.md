# MetaEx-Skill

An anchor-free **learned explorer** for text-space skill optimization on frozen language models.
At each round a frozen meta-controller reads the target's own **compact-fail** evidence (failing
trajectories only, compressed), samples a *population* of edit operations from an explicit
op-menu, authors a full skill under each, and keeps the best through a strict held-out selection
gate—steered across rounds by *linguistic feedback* on which operations helped or regressed.

This repository contains the runners for the fully-reproducible open-weights experiments
(Qwen3-8B / Qwen3.6-35B-A3B on the DSPy harness).

## Setup

```bash
pip install -r requirements.txt          # dspy, litellm, datasets, huggingface_hub<0.35
cp .env.example .env                      # then edit OLLAMA_BASE to your endpoint
```

Serve the frozen target model on any OpenAI-compatible endpoint (e.g. [ollama](https://ollama.com)):

```bash
ollama pull qwen3:8b                       # or qwen3.6:35b-a3b for the 35B target
# ollama exposes an OpenAI-compatible API at http://localhost:11434/v1
```

Point `OLLAMA_BASE` in `.env` at that endpoint. Models are served **non-thinking**
(`extra_body={"reasoning":{"effort":"none"}}`); Qwen3.5/3.6 support this natively.

## Datasets (open-model experiments)

The four public benchmarks below are **downloaded automatically** the first time a runner
touches them—no manual preparation needed:

| Benchmark | Source | Loader |
|---|---|---|
| **HotpotQA** | DSPy built-in | `dspy.datasets.HotPotQA` |
| **IFBench**  | Hugging Face | `datasets.load_dataset(...)` |
| **HoVer**    | Hugging Face `hover-nlp/hover` | `datasets.load_dataset(..., trust_remote_code=True)` |
| **PUPA**     | Hugging Face `Columbia-NLP/PUPA` (`pupa_new`) | `datasets.load_dataset(...)` |

All splits use `seed=42` (train ~150 / val ~50). The first HuggingFace pull may need
`huggingface_hub<0.35` (pinned in `requirements.txt`) for the older dataset ids.

## Running

```bash
# Iterative refinement, one method at a time (bench, method, rounds):
python gepa_refine.py hotpotqa skillopt 30
python gepa_refine.py pupa metaex 30              # (methods: oneshot skillopt trace2skill evoseed gepa)

# MetaEx op-menu population sweep (bench, M_pop, feedback_mode, rounds):
python gepa_metaex_ablation_grid.py hotpotqa 4 fail 30
#   M_pop in {1,2,4,6} (capped at the 6-op menu); mode in {full,fail,success,both}

# Anchor ablation — condition the explorer on a base generator's seed skill:
python gepa_metaex_anchor.py hotpotqa 4 fail 30 gepa   # anchor in {none,evoseed,skillopt,trace2skill,gepa}

# Cross-model transfer — apply an authored skill zero-shot to another frozen target:
SKILLS_DIR=./skills python gepa_transfer.py pupa ./skills/pupa_metaex.md
```

Authored skills are saved to `SKILLS_DIR` (default `./skills`).

## Closed-model benchmarks (separate harness)

The paper also reports a broad closed-model validation on six benchmarks spanning direct-chat,
tool-use, vision, and embodied harnesses. **Those harnesses are not included in this
repository** (they wrap tool loops, image inputs, and a TextWorld environment). Their public
data sources are listed for reference:

| Benchmark | Task / harness | Source |
|---|---|---|
| **SearchQA** | extractive QA, single-round | Dunn et al., 2017 |
| **SpreadsheetBench** | cell edits, tool-loop | Ma et al., 2024 |
| **LiveMathematicianBench** | research-level math MCQ | 2026 |
| **DocVQA** | document VQA, vision | Mathew et al., 2021 |
| **OfficeQA Pro** | numeric grounded reasoning | 2026 |
| **ALFWorld** | embodied multi-step (TextWorld) | Shridhar et al., 2020 |

See the paper appendix for the closed-model protocol and per-benchmark configuration.
