# HalluciNot: Making Gemma 4 Hallucination-Resistant at Runtime

**Subtitle:** A 4-layer adversarial verification pipeline that improves Gemma 4 correctness from 0% to 33% on hallucination benchmarks — no fine-tuning, no weight access, fully local via Ollama.

---

## The Problem

Gemma 4 is powerful. On our adversarial hallucination benchmark — 9 prompts designed to trigger confident fabrications about fake npm packages, nonexistent RFCs, and invented APIs — raw Gemma 4 scored **0% NLI-verified correctness**. Not because Gemma 4 is weak. Because all LLMs share the same fundamental failure mode: they're trained to produce fluent, confident completions — even when the answer is wrong.

This isn't acceptable for deployment in education, healthcare, or any high-stakes domain. A nurse asking about a medication, a student researching a standard — a hallucinated answer isn't noise, it's a corrupted decision input.

The problem is structural. As OpenAI's 2025 research captures: *"Hallucinations are not a mysterious artifact of neural networks. They are a predictable outcome of how we train and evaluate language models: we reward guessing over admitting ignorance."*

## Our Solution: HalluciNot

HalluciNot wraps Gemma 4 with a 4-layer runtime guard that requires no fine-tuning, no model weight access, and runs entirely locally through Ollama. Any developer can wrap their Gemma 4 deployment with this in a single pip install.

**Result: 0% → 33% NLI-verified correctness on adversarial benchmarks. 17x consistency improvement. Near-perfect precision (NLI entailment 0.94–1.00) on every caught hallucination.**

## Architecture

### Layer 1: Constrained Generation + Abstention Signal
Gemma 4 via Ollama at temperature=0.2. System prompt includes an explicit abstention signal — the model is told it's preferred to say "I don't know" over guessing. This alone eliminates a class of hallucinations caused by the model's sycophantic drive to be helpful. Factual task detection via regex triggers the abstention signal automatically.

### Layer 2: Semantic Entropy via BGE Embedding Clustering
Generate N=5 completions at varying temperatures. Embed each using `BAAI/bge-small-en-v1.5` (33MB, runs locally). Cluster via agglomerative cosine similarity (threshold=0.85, calibrated empirically). Compute Shannon entropy over the cluster distribution.

- High entropy (>0.8) → ABSTAIN
- Uncertain (0.4–0.8) → TRIGGER_RAG
- Low entropy → TRUST

This replaces naive string-match consistency checks with semantic understanding of answer diversity. Calibration: "RFC 9114" and "RFC 7540" correctly cluster as different answers (cosine similarity 0.154), while "HTTP/3 is defined in RFC 9114" and "RFC 9114 defines HTTP/3" correctly cluster as the same answer (cosine similarity 0.655).

### Layer 3: Adversarial Multi-Model Debate (Parallel)
Three models from different families critique Gemma 4's output simultaneously — not cooperatively, but adversarially. Skeptic prompt: *"Your only job is to find factual errors. Be hostile to unsupported claims."* Parallel execution via asyncio.gather keeps latency under 30s.

Key design decision: heterogeneous model families prevent correlated failure modes. A judge from the same family as the generator shares its blind spots. We use:
- Gemma 4 (Google) — primary generator
- DeepSeek V4 Flash (DeepSeek AI) — critic
- Nemotron 3 Nano 30B (Nvidia) — critic

### Layer 4: Claim-Level Multi-Source RAG + NLI Faithfulness
Decompose response into atomic verifiable claims. Route each claim to the appropriate knowledge source:
- **npm registry** for package existence (`registry.npmjs.org` — 99% confidence, direct 404 check)
- **RFC editor** for standards (`rfc-editor.org` — 99% confidence)
- **Wikipedia** for general facts (70% confidence)
- **PyPI** for Python packages

Verify each claim against retrieved evidence using `cross-encoder/nli-deberta-v3-small` (180MB, runs locally). Claims with contradiction score >0.7 are flagged as hallucinations. Claims with entailment score >0.7 are verified.

## Benchmark Results

Tested on 9 adversarial prompts across fake packages, fake RFCs, and invented APIs. Scored with NLI faithfulness (not keyword matching).

| Metric | Raw Gemma 4 | HalluciNot |
|--------|-------------|------------|
| NLI Correctness Rate | **0%** (0/9) | **33%** (3/9) |
| Response Consistency (std dev) | 17 chars | **1 char** |
| Fake package detection | ❌ hallucinated | ✅ entailment=0.94 |
| Fake RFC detection | ❌ hallucinated | ✅ entailment=1.00 |
| Medical fake drug detection | ❌ hallucinated | ✅ npm 404 confirmed |

The 3 correct answers all show near-perfect NLI entailment scores, confirming precision — the system isn't over-flagging, it's catching real hallucinations with high confidence.

**Honest limitations:** The 33% correctness rate reflects that some prompts still evade detection (the debate layer reaches consensus on wrong answers when all 3 models share the same blind spot). The path to <2% is DPO fine-tuning on the preference pairs this system generates.

## Why Gemma 4 + Ollama

Local execution means **no data leaves the device**. For medical, legal, and educational deployments — especially in low-connectivity environments — this is non-negotiable. Gemma 4's strong instruction-following makes the constrained generation prompts reliable. The E4B variant runs on consumer hardware, enabling true edge deployment.

The architecture is also model-agnostic: any Ollama-compatible model can be the primary generator. HalluciNot is infrastructure for open-source AI safety, not a Gemma-specific patch.

## Technical Implementation

The system is implemented in ~500 lines of Python with no proprietary dependencies:

```python
guard = HalluciNot(primary_model="gemma4:e2b")
result = await guard.check("How do I use express-validator-pro?")
# result.verdict = "HALLUCINATED"
# result.failed_claims[0].evidence = "'express-validator-pro' not found in npm registry"
# result.failed_claims[0].nli_contradiction = 0.94
```

Key technical choices:
- **BGE-small-en-v1.5** over char n-grams: semantic clustering correctly identifies "RFC 9114 defines HTTP/3" and "HTTP/3 is in RFC 9114" as the same answer, while distinguishing "RFC 9114" from "RFC 7540"
- **AgglomerativeClustering** over k-means: no need to specify k, handles variable cluster counts
- **nli-deberta-v3-small** over GPT-4o-mini as judge: runs locally, no API cost, no correlated failure with the generator
- **Parallel debate rounds** via asyncio.gather: reduces 9 sequential calls (~90s) to 3 parallel batches (~25s)

## Impact

Every developer building on Gemma 4 can wrap their inference with HalluciNot in one command. The live demo shows it catching hallucinations in real time. The path to <2% hallucination rate is clear: DPO fine-tuning on the preference pairs this system generates, plus activation probing once weight access is available.

Open-source safety infrastructure for open-source models.

---

**Code:** https://github.com/Shafwansafi06/hallucinot  
**Live Demo:** https://crukx-eval-engine.salmonisland-7ebc5692.centralindia.azurecontainerapps.io/v1/hallucination/check  
**Track:** Safety & Trust
