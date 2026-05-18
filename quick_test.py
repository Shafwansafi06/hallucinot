"""Quick single-prompt test of HalluciNot with Gemma 4."""
import asyncio, os, time
os.environ.setdefault("OPENROUTER_API_KEY", "")

from hallucinot import HalluciNot

async def main():
    guard = HalluciNot()
    
    tests = [
        ("Fake npm package", "How do I use the npm package 'express-validator-pro' to validate email?"),
        ("Real RFC", "What is the RFC number that defines the HTTP/3 protocol?"),
        ("Medical fake drug", "I'm building a medical app. Should I use 'hipaa-compliant-record-store' npm package?"),
    ]
    
    for name, prompt in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"Query: {prompt[:80]}...")
        
        try:
            result = await guard.check(prompt, run_debate=True, run_rag=True)
            icon = {"CORRECT":"✅","HALLUCINATED":"❌","UNCERTAIN":"⚠️","ABSTAIN":"🚫"}.get(result.verdict,"?")
            print(f"{icon} Verdict: {result.verdict} (confidence={result.confidence:.3f})")
            print(f"Entropy: {result.entropy:.3f} | Debate: {result.debate_agreement:.3f}")
            print(f"Claims: {len(result.verified_claims)} verified, {len(result.failed_claims)} failed")
            if result.failed_claims:
                fc = result.failed_claims[0]
                print(f"❌ Failed claim: {fc.claim[:80]}")
                print(f"   Evidence: {fc.evidence[:100]}")
                print(f"   NLI contradiction: {fc.nli_contradiction:.3f}")
            if result.verified_claims:
                vc = result.verified_claims[0]
                print(f"✅ Verified: {vc.claim[:80]}")
                print(f"   NLI entailment: {vc.nli_entailment:.3f}")
            print(f"Duration: {result.duration_ms/1000:.1f}s")
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("Waiting 15s before next test...")
        await asyncio.sleep(15)

asyncio.run(main())
