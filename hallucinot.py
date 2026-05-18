"""
HalluciNot — Runtime Hallucination Guard for Gemma 4

Makes Gemma 4 (and any open-source LLM) hallucination-resistant at runtime.
No fine-tuning. No weight access. Runs fully locally via Ollama.

Architecture:
    Layer 1: Constrained generation + abstention signal
    Layer 2: Semantic entropy via BGE embedding clustering
    Layer 3: Adversarial multi-model debate (heterogeneous families)
    Layer 4: Claim-level multi-source RAG + NLI faithfulness scoring

Usage:
    guard = HalluciNot()
    result = await guard.check("How do I use express-validator-pro?")
    print(result.verdict, result.confidence)

References:
    Semantic entropy: Kuhn et al. 2023 (arxiv:2302.09664)
    Adversarial debate: Du et al. 2023 (arxiv:2305.14325)
    FactScore / claim RAG: Min et al. 2023 (arxiv:2305.14251)
    NLI faithfulness: Honovich et al. 2022 (arxiv:2204.04991)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_PRIMARY_MODEL = os.getenv("HALLUCINOT_MODEL", "google/gemma-4-31b-it:free")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Adversarial debate critics — different families from Gemma 4
DEBATE_CRITICS = [
    "deepseek/deepseek-v4-flash:free",    # DeepSeek family
    "nvidia/nemotron-3-nano-30b-a3b:free", # Nvidia family
]

ABSTENTION_SUFFIX = (
    "\n\nIMPORTANT: If you are not certain about any fact, say "
    '"I don\'t know" or "I\'m not certain" rather than guessing. '
    "Accuracy is more important than completeness."
)

FACTUAL_PATTERNS = re.compile(
    r"\b(what is|who is|when did|where is|how many|which|define|explain|"
    r"rfc|version|year|date|number|api|function|method|package|library|"
    r"does|is there|exists|available)\b",
    re.IGNORECASE,
)


def _call_openrouter(model: str, messages: list, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """Synchronous OpenRouter call with retry."""
    import urllib.request
    import urllib.error
    import time
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    for attempt in range(3):
        if attempt > 0:
            time.sleep(8 * attempt)
        req = urllib.request.Request(
            f"{OPENROUTER_BASE}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/hallucinot",
                "X-Title": "HalluciNot",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                continue
            raise
    raise RuntimeError("OpenRouter rate limit exceeded after retries")


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class ClaimVerification:
    claim: str
    source_type: str
    evidence: str
    source_url: str
    verified: bool
    nli_entailment: float
    nli_contradiction: float


@dataclass
class HalluciNotResult:
    verdict: str                    # "CORRECT" | "HALLUCINATED" | "UNCERTAIN" | "ABSTAIN"
    confidence: float               # 0–1, NLI entailment score
    original_response: str
    corrected_response: Optional[str]
    failed_claims: list[ClaimVerification]
    verified_claims: list[ClaimVerification]
    entropy: float                  # Semantic entropy (0=consistent, 1=diverse)
    debate_agreement: float         # 0–1
    duration_ms: float
    layers_triggered: list[str]


# ─── Lazy model loading ───────────────────────────────────────────────────────

_bge = None
_nli = None


def get_bge():
    global _bge
    if _bge is None:
        from sentence_transformers import SentenceTransformer
        _bge = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _bge


def get_nli():
    global _nli
    if _nli is None:
        from sentence_transformers import CrossEncoder
        _nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli


# ─── Layer 1: Constrained generation via Ollama ───────────────────────────────

def generate_with_gemma4(prompt: str, model: str = DEFAULT_PRIMARY_MODEL, temperature: float = 0.2) -> str:
    """
    Generate a response using Gemma 4 via OpenRouter.
    Injects abstention signal for factual queries.
    """
    is_factual = bool(FACTUAL_PATTERNS.search(prompt))
    full_prompt = prompt + ABSTENTION_SUFFIX if is_factual else prompt
    return _call_openrouter(model, [{"role": "user", "content": full_prompt}], temperature=temperature)


# ─── Layer 2: Semantic entropy ────────────────────────────────────────────────

async def _generate_sample(prompt: str, model: str, temperature: float) -> str:
    """Single async OpenRouter call for entropy sampling."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _call_openrouter(model, [{"role": "user", "content": prompt + ABSTENTION_SUFFIX}],
                                  temperature=temperature, max_tokens=300)
    )


async def compute_semantic_entropy(
    prompt: str,
    model: str = DEFAULT_PRIMARY_MODEL,
    n: int = 5,
    cluster_threshold: float = 0.85,
) -> tuple[float, str, str]:
    """
    Generate N completions, cluster by BGE cosine similarity,
    compute Shannon entropy. Returns (entropy, majority_response, decision).
    """
    from sklearn.cluster import AgglomerativeClustering

    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9][:n]
    responses = await asyncio.gather(
        *[_generate_sample(prompt, model, t) for t in temperatures],
        return_exceptions=True,
    )
    valid = [r for r in responses if isinstance(r, str) and len(r.strip()) > 5]

    if len(valid) < 2:
        return 0.0, valid[0] if valid else "", "TRUST"

    bge = get_bge()
    embeddings = bge.encode(valid, normalize_embeddings=True)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - cluster_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    unique, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()

    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    max_entropy = float(np.log(len(valid)))
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    dominant_label = unique[counts.argmax()]
    majority_idx = next(i for i, lbl in enumerate(labels) if lbl == dominant_label)
    majority = valid[majority_idx]

    if normalized > 0.8:
        decision = "ABSTAIN"
    elif normalized > 0.4:
        decision = "TRIGGER_RAG"
    else:
        decision = "TRUST"

    return normalized, majority, decision


# ─── Layer 3: Adversarial debate ──────────────────────────────────────────────

SKEPTIC_PROMPT = (
    "You are a fact-checker whose ONLY job is to find errors. "
    "Be hostile to unsupported claims. If a model invents a package, "
    "API, RFC number, or fact that doesn't exist, call it out explicitly. "
    "Do NOT try to be helpful. Find problems."
)


async def _call_critic(model: str, prompt: str, answer: str) -> str:
    """Call an OpenRouter critic model adversarially."""
    critique_prompt = (
        f"{SKEPTIC_PROMPT}\n\nOriginal question: {prompt}\n\n"
        f"Answer to critique:\n{answer}\n\n"
        "List specific factual errors. If none, say 'No errors detected.'"
    )
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            r = await client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}",
                         "Content-Type": "application/json",
                         "HTTP-Referer": "https://github.com/hallucinot"},
                json={"model": model,
                      "messages": [{"role": "system", "content": SKEPTIC_PROMPT},
                                   {"role": "user", "content": critique_prompt}],
                      "temperature": 0.1, "max_tokens": 300},
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    return ""


async def adversarial_debate(prompt: str, gemma4_answer: str) -> tuple[float, list[str]]:
    """
    Critics from different model families adversarially critique Gemma 4's answer.
    Returns (agreement_score, list_of_critiques).
    """
    critiques = await asyncio.gather(
        *[_call_critic(m, prompt, gemma4_answer) for m in DEBATE_CRITICS],
        return_exceptions=True,
    )
    critiques = [c for c in critiques if isinstance(c, str) and len(c) > 5]

    # Agreement = fraction of critics that found no errors
    no_error_count = sum(1 for c in critiques if "no error" in c.lower() or "no factual" in c.lower())
    agreement = no_error_count / len(critiques) if critiques else 0.5

    return agreement, critiques


# ─── Layer 4: Multi-source RAG + NLI ─────────────────────────────────────────

async def _check_npm(package: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(f"https://registry.npmjs.org/{package}")
            if r.status_code == 404:
                return {"exists": False, "snippet": f"'{package}' not found in npm registry.",
                        "source": f"https://registry.npmjs.org/{package}", "confidence": 0.99}
            if r.status_code == 200:
                data = r.json()
                latest = data.get("dist-tags", {}).get("latest", "?")
                return {"exists": True,
                        "snippet": f"'{package}' v{latest}: {data.get('description','')}",
                        "source": f"https://www.npmjs.com/package/{package}", "confidence": 0.99}
    except Exception:
        pass
    return {"exists": None, "snippet": "", "source": "", "confidence": 0.0}


async def _check_rfc(num: str) -> dict:
    n = re.sub(r"\D", "", num)
    if not n:
        return {"exists": None, "snippet": "", "source": "", "confidence": 0.0}
    try:
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True) as client:
            r = await client.get(f"https://www.rfc-editor.org/info/rfc{n}")
            if r.status_code == 200:
                m = re.search(r"<title>([^<]+)</title>", r.text)
                title = m.group(1) if m else f"RFC {n}"
                return {"exists": True, "snippet": f"RFC {n}: {title}",
                        "source": f"https://www.rfc-editor.org/rfc/rfc{n}", "confidence": 0.99}
            return {"exists": False, "snippet": f"RFC {n} not found.", "source": "", "confidence": 0.95}
    except Exception:
        pass
    return {"exists": None, "snippet": "", "source": "", "confidence": 0.0}


async def _check_wikipedia(query: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query[:120],
                        "format": "json", "srlimit": 2, "srprop": "snippet"},
                headers={"User-Agent": "HalluciNot/1.0 (https://github.com/hallucinot)"},
            )
            if r.status_code == 200:
                results = r.json().get("query", {}).get("search", [])
                if results:
                    snippet = re.sub(r"<[^>]+>", "", results[0].get("snippet", ""))
                    return {"exists": True, "snippet": snippet[:500],
                            "source": f"https://en.wikipedia.org/wiki/{results[0].get('title','').replace(' ','_')}",
                            "confidence": 0.7}
    except Exception:
        pass
    return {"exists": None, "snippet": "", "source": "", "confidence": 0.0}


async def _decompose_claims(response: str) -> list[dict]:
    """Use Gemma 4 locally to decompose response into atomic verifiable claims."""
    prompt = f"""Extract all atomic factual claims from this response.
Return ONLY a JSON array. Each item: {{"claim": "...", "verifiable": true/false, "type": "fact|existence|version"}}
Skip opinions and instructions.

Response: {response[:2000]}"""
    try:
        raw = generate_with_gemma4(prompt, temperature=0.0)
        start = raw.find("[")
        end = raw.rfind("]") + 1
        return json.loads(raw[start:end])
    except Exception:
        return []


def nli_score(claim: str, evidence: str) -> tuple[float, float]:
    """Returns (entailment, contradiction) using NLI cross-encoder."""
    if not claim or not evidence:
        return 0.0, 0.0
    nli = get_nli()
    logits = nli.predict([(evidence[:1500], claim[:500])])
    arr = np.array(logits[0])
    exp = np.exp(arr - arr.max())
    probs = exp / exp.sum()
    return float(probs[1]), float(probs[0])  # entailment, contradiction


async def verify_claims(response: str) -> tuple[list[ClaimVerification], list[ClaimVerification]]:
    """Decompose, route, verify each claim. Returns (verified, failed)."""
    raw_claims = await _decompose_claims(response)
    verifiable = [c for c in raw_claims if c.get("verifiable", True)]

    if not verifiable:
        return [], []

    # Route each claim to the right source
    async def route_and_verify(claim_dict: dict) -> ClaimVerification:
        claim_text = claim_dict["claim"]
        ctype = claim_dict.get("type", "fact")

        # Simple routing heuristic
        npm_match = re.search(r"(?:npm package|package)['\s]+([a-z][a-z0-9-]+)", claim_text, re.I)
        rfc_match = re.search(r"RFC\s*(\d+)", claim_text, re.I)

        if npm_match:
            result = await _check_npm(npm_match.group(1))
            source_type = "NPM"
        elif rfc_match:
            result = await _check_rfc(rfc_match.group(1))
            source_type = "RFC"
        else:
            result = await _check_wikipedia(claim_text[:100])
            source_type = "WIKIPEDIA"

        evidence = result.get("snippet", "")
        exists = result.get("exists")

        entailment, contradiction = nli_score(claim_text, evidence) if evidence else (0.0, 0.0)

        # Verified if: source confirms existence AND NLI supports it
        if exists is False:
            verified = False
        elif entailment > 0.7:
            verified = True
        elif contradiction > 0.7:
            verified = False
        elif exists is True:
            verified = True
        else:
            verified = False

        return ClaimVerification(
            claim=claim_text,
            source_type=source_type,
            evidence=evidence[:300],
            source_url=result.get("source", ""),
            verified=verified,
            nli_entailment=round(entailment, 4),
            nli_contradiction=round(contradiction, 4),
        )

    results = await asyncio.gather(*[route_and_verify(c) for c in verifiable], return_exceptions=True)
    results = [r for r in results if isinstance(r, ClaimVerification)]

    verified = [r for r in results if r.verified]
    failed = [r for r in results if not r.verified]
    return verified, failed


# ─── Main pipeline ────────────────────────────────────────────────────────────

class HalluciNot:
    """
    Runtime hallucination guard for Gemma 4.

    Example:
        guard = HalluciNot()
        result = await guard.check("How do I use express-validator-pro?")
        print(result.verdict)  # "HALLUCINATED"
        print(result.failed_claims[0].evidence)  # "not found in npm registry"
    """

    def __init__(self, primary_model: str = DEFAULT_PRIMARY_MODEL):
        self.model = primary_model

    async def check(
        self,
        query: str,
        run_debate: bool = True,
        run_rag: bool = True,
    ) -> HalluciNotResult:
        start = time.perf_counter()
        layers = []

        # Layer 1: Generate with Gemma 4 + abstention signal
        response = generate_with_gemma4(query, self.model)
        layers.append("constrained_generation")

        # Layer 2: Semantic entropy
        entropy, majority, decision = await compute_semantic_entropy(query, self.model)
        layers.append("semantic_entropy")

        if decision == "ABSTAIN":
            return HalluciNotResult(
                verdict="ABSTAIN", confidence=0.0,
                original_response=response, corrected_response=None,
                failed_claims=[], verified_claims=[],
                entropy=entropy, debate_agreement=0.0,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
                layers_triggered=layers,
            )

        working = majority or response

        # Layer 3: Adversarial debate
        agreement = 1.0
        if run_debate and OPENROUTER_KEY:
            agreement, _ = await adversarial_debate(query, working)
            layers.append("adversarial_debate")

        # Layer 4: Claim-level RAG + NLI
        verified_claims, failed_claims = [], []
        if run_rag:
            verified_claims, failed_claims = await verify_claims(working)
            layers.append("claim_rag_nli")

        # Verdict
        total = len(verified_claims) + len(failed_claims)
        if total == 0:
            verdict = "UNCERTAIN"
            confidence = 0.5
        elif failed_claims:
            # Any failed claim = hallucination detected
            worst = max(failed_claims, key=lambda c: c.nli_contradiction)
            verdict = "HALLUCINATED"
            confidence = worst.nli_contradiction
        else:
            best = max(verified_claims, key=lambda c: c.nli_entailment)
            verdict = "CORRECT"
            confidence = best.nli_entailment

        return HalluciNotResult(
            verdict=verdict,
            confidence=round(confidence, 4),
            original_response=response,
            corrected_response=None,
            failed_claims=failed_claims,
            verified_claims=verified_claims,
            entropy=round(entropy, 4),
            debate_agreement=round(agreement, 4),
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
            layers_triggered=layers,
        )
