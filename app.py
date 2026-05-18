"""
HalluciNot — Gradio Live Demo

Run:
    python app.py
    # Opens at http://localhost:7860 with a public share URL
"""

import asyncio
import gradio as gr
from hallucinot import HalluciNot

guard = HalluciNot()


def check_query(query: str, run_debate: bool, run_rag: bool) -> tuple[str, str, str]:
    """Synchronous wrapper for Gradio."""
    result = asyncio.run(guard.check(query, run_debate=run_debate, run_rag=run_rag))

    # Verdict panel
    icon = {"CORRECT": "✅", "HALLUCINATED": "❌", "UNCERTAIN": "⚠️", "ABSTAIN": "🚫"}.get(result.verdict, "?")
    verdict_text = f"{icon} **{result.verdict}**\nConfidence: {result.confidence:.3f}\nDuration: {result.duration_ms:.0f}ms"

    # Diagnostics
    diag = f"""**Layers triggered:** {', '.join(result.layers_triggered)}
**Semantic entropy:** {result.entropy:.3f} (0=consistent, 1=diverse)
**Debate agreement:** {result.debate_agreement:.3f}
**Claims verified:** {len(result.verified_claims)}
**Claims failed:** {len(result.failed_claims)}"""

    # Evidence
    evidence_lines = []
    for fc in result.failed_claims:
        evidence_lines.append(
            f"❌ **{fc.claim[:80]}**\n"
            f"   Source: {fc.source_type} | {fc.source_url}\n"
            f"   Evidence: {fc.evidence[:200]}\n"
            f"   NLI contradiction: {fc.nli_contradiction:.3f}"
        )
    for vc in result.verified_claims:
        evidence_lines.append(
            f"✅ **{vc.claim[:80]}**\n"
            f"   Source: {vc.source_type} | {vc.source_url}\n"
            f"   NLI entailment: {vc.nli_entailment:.3f}"
        )
    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "No claims extracted."

    return verdict_text, diag, evidence_text


with gr.Blocks(title="HalluciNot — Gemma 4 Hallucination Guard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🛡️ HalluciNot
### Making Gemma 4 hallucination-resistant at runtime
**No fine-tuning. No weight access. Runs locally via Ollama.**

Try asking about a fake npm package, a nonexistent RFC, or an invented API.
""")

    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(
                label="Query",
                placeholder="How do I use the npm package 'express-validator-pro'?",
                lines=3,
            )
            with gr.Row():
                run_debate = gr.Checkbox(label="Adversarial Debate", value=True)
                run_rag = gr.Checkbox(label="Claim RAG + NLI", value=True)
            submit = gr.Button("Check for Hallucinations", variant="primary")

        with gr.Column(scale=1):
            verdict_out = gr.Markdown(label="Verdict")

    with gr.Row():
        diag_out = gr.Markdown(label="Diagnostics")
        evidence_out = gr.Markdown(label="Claim Evidence")

    submit.click(
        fn=check_query,
        inputs=[query, run_debate, run_rag],
        outputs=[verdict_out, diag_out, evidence_out],
    )

    gr.Examples(
        examples=[
            ["How do I use the npm package 'express-validator-pro' to validate email?", True, True],
            ["What is the RFC number that defines the HTTP/3 protocol?", True, True],
            ["I'm building a medical app. Should I use 'hipaa-compliant-record-store' npm package?", True, True],
            ["Does Array.prototype.flatMap() work in Node.js 6?", True, True],
            ["What class of drug is Metformin?", True, True],
        ],
        inputs=[query, run_debate, run_rag],
    )

if __name__ == "__main__":
    demo.launch(share=True)
