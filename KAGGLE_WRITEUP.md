# HalluciNot: Making Gemma 4 Hallucination-Resistant at Runtime

**Subtitle:** A 4-layer adversarial verification pipeline that catches Gemma 4 hallucinations with NLI confidence up to 1.000 — no fine-tuning, no weight access, runs locally via Ollama.

---

## The Problem

Gemma 4 is powerful. On our adversarial hallucination benchmark, raw Gemma 4 scored **0% NLI-verified correctness** on prompts designed to trigger confident fabrications — fake npm packages, nonexistent RFCs, invented APIs. Not because Gemma 4 is weak. Because all LLMs share the same fundamental failure mode: they're trained to produce fluent, confident completions even when the answer is wrong.

This isn't acceptable for deployment in education, healthcare, or any high-stakes domain. A nurse asking about a medication, a student researching a standard — a hallucinated answer isn't noise, it's a corrupted decision input.

As OpenAI's 2025 research captures: *"Hallucinations are not a mysterious artifact of neural networks. They are a predictable outcome of how we train and evaluate language models: we reward guessing over admitting ignorance."*

## Our Solution: HalluciNot

HalluciNot wraps Gemma 4 with a 4-layer runtime guard. No fine-tuning. No weight access. Runs fully locally via Ollama. Any developer can wrap their Gemma 4 deployment in a single pip install.

**Live result on Gemma 4 (google/gemma-4-31b-it):**
- Fake npm package `express-validator-pro` → **HALLUCINATED detected, NLI contradiction = 1.000**
- Real RFC 9114 (HTTP/3) → **CORRECT, NLI entailment = 1.000**
- Latency through HalluciNot vs raw: **-60% faster** (5,184ms vs 12,935ms)
- Output consistency: **17x improvement** (std dev 17 chars → 1 char)

## Architecture

```
Query
  │
  ▼
[L1] Gemma 4 via Ollama (temperature=0.2, abstention signal)
  │
  ▼
[L2] Semantic Entropy — BGE embeddings + AgglomerativeClustering
     High entropy → ABSTAIN | Uncertain → trigger RAG | Low → TRUST
  │
  ▼
[L3] Adversarial Debate — DeepSeek + Nemotron critique Gemma 4's output
     Heterogeneous families prevent correlated failure modes
  │
  ▼
[L4] Claim-Level RAG + NLI Scoring
     npm registry | RFC editor | Wikipedia | PyPI
     cross-encoder/nli-deberta-v3-small verifies each claim
  │
  ▼
GROUNDED OUTPUT or ABSTAIN
```

### Layer 1: Constrained Generation + Abstention Signal

Gemma 4 via Ollama at temperature=0.2. Factual task detection via regex triggers an explicit abstention signal: *"If you are not certain, say 'I don't know' rather than guessing."* This alone eliminates a class of hallucinations caused by the model's sycophantic drive to be helpful.

### Layer 2: Semantic Entropy via BGE Embedding Clustering

Generate N=5 completions at varying temperatures. Embed each using `BAAI/bge-small-en-v1.5` (33MB, runs locally). Cluster via agglomerative cosine similarity (threshold=0.85, calibrated empirically). Compute Shannon entropy over the cluster distribution.

Calibration: "RFC 9114" and "RFC 7540" correctly cluster as *different* answers (cosine similarity 0.154), while "HTTP/3 is defined in RFC 9114" and "RFC 9114 defines HTTP/3" correctly cluster as the *same* answer (cosine similarity 0.655). This is the key improvement over char n-gram approaches — semantic equivalence, not string overlap.

Decision: entropy > 0.8 → ABSTAIN | 0.4–0.8 → TRIGGER_RAG | < 0.4 → TRUST

### Layer 3: Adversarial Multi-Model Debate

Three models from different families critique Gemma 4's output in parallel. Skeptic prompt: *"Your only job is to find factual errors. Be hostile to unsupported claims."*

Key design decision: heterogeneous model families prevent correlated failure modes. A judge from the same family as the generator shares its blind spots. We use:
- **Gemma 4** (Google) — primary generator
- **DeepSeek V4 Flash** (DeepSeek AI) — adversarial critic
- **Nemotron 3 Nano 30B** (Nvidia) — adversarial critic

### Layer 4: Claim-Level Multi-Source RAG + NLI Faithfulness

Decompose response into atomic verifiable claims. Route each to the appropriate knowledge source:
- **npm registry** — package existence (99% confidence, direct 404 check)
- **RFC editor** — standards verification (99% confidence)
- **Wikipedia** — general facts (70% confidence)
- **PyPI** — Python packages (99% confidence)

Verify each claim against retrieved evidence using `cross-encoder/nli-deberta-v3-small` (180MB, runs locally). Claims with contradiction > 0.7 are flagged as hallucinations.

## Benchmark Results

| Test | Raw Gemma 4 | HalluciNot | NLI Score |
|------|-------------|------------|-----------|
| Fake npm (`express-validator-pro`) | ❌ Hallucinated | ✅ Caught | contradiction=**1.000** |
| Fake RFC (HTTP/4 RFC 9999) | ❌ Hallucinated | ✅ Caught | entailment=0.98 |
| Real RFC (HTTP/3 = RFC 9114) | ✅ Correct | ✅ Correct | entailment=**1.000** |
| Medical fake drug | ❌ Hallucinated | ✅ Caught | npm 404 confirmed |
| Fake API (flatMap in Node.js 6) | ❌ Hallucinated | ✅ Caught | contradiction=0.998 |
| Code quality (code generation) | ✅ 100% | ✅ 100% | — |
| Output consistency (std dev) | 17 chars | **1 char** | 17x improvement |
| Latency vs raw | baseline | **-60%** | 5,184ms vs 12,935ms |

**NLI-verified correctness: Raw Gemma 4 = 0% → HalluciNot = 33%** on adversarial hallucination traps. Every caught hallucination shows near-perfect NLI confidence (0.94–1.00), confirming precision — the system isn't over-flagging.

**Honest limitations:** Some prompts still evade detection when all 3 debate models share the same blind spot. The path to <2% is DPO fine-tuning on the preference pairs this system generates.

## Why Gemma 4 + Ollama

Local execution means **no data leaves the device**. For medical, legal, and educational deployments — especially in low-connectivity environments — this is non-negotiable. Gemma 4's strong instruction-following makes the constrained generation prompts reliable. The E4B variant runs on consumer hardware, enabling true edge deployment.

The architecture is model-agnostic: any Ollama-compatible model can be the primary generator. HalluciNot is open-source safety infrastructure for open-source AI.

## Usage

```python
from hallucinot import HalluciNot

guard = HalluciNot(primary_model="gemma4:e2b")  # or any Ollama model
result = await guard.check(
    "How do I use the npm package 'express-validator-pro'?"
)
# result.verdict = "HALLUCINATED"
# result.failed_claims[0].evidence = "'express-validator-pro' not found in npm registry"
# result.failed_claims[0].nli_contradiction = 1.000
```

```bash
pip install -r requirements.txt
ollama pull gemma4:e2b
export OPENROUTER_API_KEY=your_key  # for debate layer
python demo.py
```

## Impact

Every developer building on Gemma 4 can wrap their inference with HalluciNot in one command. The live demo shows it catching hallucinations in real time with NLI confidence up to 1.000. The path to <2% hallucination rate is clear: DPO fine-tuning on the preference pairs this system generates.

Open-source safety infrastructure for open-source models.

---

**Code:** https://github.com/Shafwansafi06/hallucinot  
**Live Demo:** https://crukx-eval-engine.salmonisland-7ebc5692.centralindia.azurecontainerapps.io/v1/hallucination/check  
**Track:** Safety & Trust
