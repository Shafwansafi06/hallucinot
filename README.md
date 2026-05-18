# 🛡️ HalluciNot

**Making Gemma 4 hallucination-resistant at runtime — no fine-tuning, fully local.**

[![Safety & Trust](https://img.shields.io/badge/Track-Safety%20%26%20Trust-blue)]()
[![Ollama](https://img.shields.io/badge/Runs%20on-Ollama-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## The Problem

Raw Gemma 4 scored **0% NLI-verified correctness** on adversarial hallucination benchmarks — confidently inventing npm packages, RFC numbers, and APIs that don't exist. In medical, legal, or educational deployments, a hallucinated answer isn't noise — it's a corrupted decision input.

## The Solution

HalluciNot wraps Gemma 4 with a 4-layer runtime guard:

```
Query
  │
  ▼
[L1] Gemma 4 via Ollama (temperature=0.2, abstention signal)
  │
  ▼
[L2] Semantic Entropy — BGE embeddings + agglomerative clustering
     High entropy → ABSTAIN | Uncertain → trigger RAG | Low → TRUST
  │
  ▼
[L3] Adversarial Debate — DeepSeek + Nemotron critique Gemma 4's output
     Heterogeneous families prevent correlated failure modes
  │
  ▼
[L4] Claim-Level RAG + NLI Scoring
     npm registry | RFC editor | Wikipedia | PyPI
     NLI cross-encoder verifies each claim against evidence
  │
  ▼
GROUNDED OUTPUT or ABSTAIN
```

## Results

| Metric | Raw Gemma 4 | HalluciNot |
|--------|-------------|------------|
| NLI Correctness Rate | **0%** (0/9) | **33%** (3/9) |
| Consistency (std dev) | 17 chars | **1 char** |
| Fake package detection | ❌ | ✅ entailment=0.94 |
| Fake RFC detection | ❌ | ✅ entailment=1.00 |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenRouter API key (free at openrouter.ai)
export OPENROUTER_API_KEY=your_key

# 3. Run the demo
python demo.py
```

> **Local/Offline mode:** Set `HALLUCINOT_MODEL=gemma4:e2b` and run `ollama pull gemma4:e2b`
> to run fully locally with no data leaving your device — ideal for medical/legal deployments.

## Live Demo

```bash
python app.py  # Opens Gradio UI with public share URL
```

## Architecture

### Layer 1: Constrained Generation
Gemma 4 runs locally via Ollama at temperature=0.2. An abstention signal is injected for factual queries: *"If you are not certain, say 'I don't know' rather than guessing."*

### Layer 2: Semantic Entropy (Kuhn et al. 2023)
Generate N=5 completions at varying temperatures. Embed each with `BAAI/bge-small-en-v1.5`. Cluster via agglomerative cosine similarity (threshold=0.85). Compute Shannon entropy over cluster distribution. High entropy → model is uncertain → abstain or trigger RAG.

### Layer 3: Adversarial Multi-Model Debate (Du et al. 2023)
Critics from different model families (DeepSeek, Nemotron) adversarially critique Gemma 4's output in parallel. Skeptic framing: *"Your only job is to find errors."* Heterogeneous families prevent correlated blind spots.

### Layer 4: Claim-Level RAG + NLI (Min et al. 2023, Honovich et al. 2022)
Decompose response into atomic verifiable claims. Route each to the appropriate knowledge source (npm registry for packages, RFC editor for standards, Wikipedia for general facts). Score with `cross-encoder/nli-deberta-v3-small` — entailment > 0.7 = verified, contradiction > 0.7 = hallucination detected.

## Why Gemma 4 + Ollama

Local execution means **no data leaves the device**. Critical for:
- 🏥 Medical apps where patient data is sensitive
- ⚖️ Legal tools where privilege matters
- 🎓 Educational deployments in low-connectivity environments

Gemma 4's strong instruction-following makes constrained generation prompts reliable. The E4B variant runs on consumer hardware — enabling true edge deployment.

## References

- Kuhn et al. 2023 — Semantic Uncertainty: [arxiv:2302.09664](https://arxiv.org/abs/2302.09664)
- Du et al. 2023 — Improving Factuality via Multi-Agent Debate: [arxiv:2305.14325](https://arxiv.org/abs/2305.14325)
- Min et al. 2023 — FactScore: [arxiv:2305.14251](https://arxiv.org/abs/2305.14251)
- Honovich et al. 2022 — TRUE: [arxiv:2204.04991](https://arxiv.org/abs/2204.04991)

## License

MIT — use freely, attribution appreciated.
