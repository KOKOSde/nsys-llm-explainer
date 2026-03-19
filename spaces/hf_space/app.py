from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import pandas as pd
import gradio as gr

from space_utils import SpaceBundle, analyze_path, coerce_upload_path, find_local_sample


APP_TITLE = "nsys-llm-explainer — Instant Nsight Trace Analyzer for Cloud LLM Inference"

CSS = """
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(42, 93, 142, 0.35), transparent 30%),
    radial-gradient(circle at top right, rgba(20, 104, 117, 0.22), transparent 26%),
    linear-gradient(180deg, #081018 0%, #0b111a 42%, #090e15 100%);
  color: #e6eef7;
  font-family: "Aptos", "Segoe UI", sans-serif;
}

.hero-card {
  border: 1px solid rgba(115, 145, 180, 0.28);
  border-radius: 22px;
  background: linear-gradient(135deg, rgba(14, 22, 34, 0.95), rgba(10, 14, 20, 0.92));
  box-shadow: 0 24px 70px rgba(0, 0, 0, 0.28);
  padding: 22px 24px;
  margin-bottom: 16px;
}

.hero-kicker {
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: #8fb4d9;
  font-size: 11px;
  font-weight: 700;
}

.hero-title {
  margin: 10px 0 10px;
  font-size: 34px;
  line-height: 1.05;
  font-weight: 800;
  color: #f3f8ff;
}

.hero-subtitle {
  color: #b2c5d9;
  font-size: 15px;
  line-height: 1.6;
  max-width: 980px;
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
}

.badge {
  display: inline-flex;
  align-items: center;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid rgba(137, 171, 207, 0.28);
  background: rgba(13, 21, 31, 0.82);
  color: #d8e6f5;
  font-size: 12px;
}

.upload-card {
  border: 1px solid rgba(88, 113, 143, 0.26);
  border-radius: 18px;
  background: rgba(10, 16, 24, 0.86);
  padding: 14px;
  margin-bottom: 14px;
}

.section-title {
  color: #f4f8fd;
  font-size: 16px;
  font-weight: 700;
  margin: 0 0 10px 0;
}

.gr-markdown, .prose {
  color: #e8eff7;
}

.wrap-long {
  white-space: pre-wrap;
  word-break: break-word;
}
"""

HEADER = """
<div class="hero-card">
  <div class="hero-kicker">Cloud ML trace intelligence</div>
  <div class="hero-title">nsys-llm-explainer — Instant Nsight Trace Analyzer for Cloud LLM Inference</div>
  <div class="hero-subtitle">
    Upload a `trace.sqlite` or `report.json` and get prioritized findings, NCCL/NVLink correlation, launch storm diagnosis,
    per-process breakdowns, and downloadable analysis artifacts. The same code path powers the CLI, dashboard, and this Space.
  </div>
  <div class="badge-row">
    <span class="badge">SQLite + report.json input</span>
    <span class="badge">Evidence-backed findings</span>
    <span class="badge">CSV + JSON downloads</span>
    <span class="badge">Built for cloud LLM traces</span>
  </div>
</div>
"""


def _empty_outputs(message: str) -> Tuple[Any, str, pd.DataFrame, str, str, list[str], pd.DataFrame]:
    empty_df = pd.DataFrame(columns=["section", "metric", "value"])
    empty_manifest = pd.DataFrame(columns=["artifact", "purpose", "path"])
    return (
        message,
        message,
        empty_df,
        message,
        message,
        [],
        empty_manifest,
    )


def _bundle_to_outputs(bundle: SpaceBundle) -> Tuple[Any, str, pd.DataFrame, str, str, list[str], pd.DataFrame]:
    summary_df = pd.DataFrame(bundle.summary_rows)
    manifest_df = pd.DataFrame(bundle.manifest_rows)
    bottleneck = next((row["value"] for row in bundle.summary_rows if row.get("metric") == "Top bottleneck"), "No bottleneck summary available")
    summary_markdown = [
        "### Quick read",
        "",
        "- Source: `{}` (`{}`)".format(bundle.source_path.name, bundle.source_kind),
        "- {}".format(bundle.report.get("generated_at") or "Generated time unavailable"),
        "- {}".format(bottleneck),
        "- Warnings: `{}`".format(len(bundle.report.get("warnings") or [])),
    ]
    files = [str(path) for path in bundle.artifact_paths]
    return (
        bundle.status_markdown,
        "\n".join(summary_markdown),
        summary_df,
        bundle.findings_markdown,
        bundle.markdown,
        files,
        manifest_df,
    )


def _resolve_path(uploaded: Any, sample_path: str) -> Optional[Path]:
    uploaded_path = coerce_upload_path(uploaded)
    if uploaded_path:
        return uploaded_path
    if sample_path:
        candidate = Path(sample_path)
        if candidate.exists():
            return candidate
    return None


def _run_analysis(uploaded: Any, sample_path: str) -> Tuple[Any, str, pd.DataFrame, str, str, list[str], pd.DataFrame]:
    path = _resolve_path(uploaded, sample_path)
    if not path:
        return _empty_outputs(
            "Upload a `trace.sqlite`/`.db` file or a `report.json` to generate the report. "
            "If you are using this Space as a demo, click `Load sample trace` first."
        )
    try:
        bundle = analyze_path(path)
        return _bundle_to_outputs(bundle)
    except Exception as exc:
        message = "Failed to analyze `{}`: `{}`".format(path.name, exc)
        return _empty_outputs(message)


def _build_demo(sample_path: Optional[Path]) -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE, css=CSS, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as demo:
        gr.HTML(HEADER)
        with gr.Row(elem_classes=["upload-card"]):
            with gr.Column(scale=6):
                upload = gr.File(
                    label="Upload trace or report",
                    file_count="single",
                    file_types=[".sqlite", ".db", ".json"],
                    type="filepath",
                )
            with gr.Column(scale=2, min_width=180):
                analyze_btn = gr.Button("Analyze trace", variant="primary")
            with gr.Column(scale=2, min_width=180):
                sample_btn = gr.Button(
                    "Load sample trace",
                    variant="secondary",
                    visible=bool(sample_path),
                )

        status = gr.Markdown("Upload a trace or report to begin.")
        sample_state = gr.State(str(sample_path) if sample_path else "")

        with gr.Tabs():
            with gr.Tab("Summary"):
                gr.Markdown("### Summary")
                summary = gr.Markdown(elem_classes=["wrap-long"])
                summary_table = gr.Dataframe(
                    headers=["section", "metric", "value"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    label="Key metrics",
                )
            with gr.Tab("Findings"):
                findings = gr.Markdown(elem_classes=["wrap-long"])
            with gr.Tab("Markdown"):
                report_markdown = gr.Markdown(elem_classes=["wrap-long"])
            with gr.Tab("Downloads"):
                gr.Markdown(
                    "### Generated artifacts\n"
                    "The analysis writes `report.md`, `report.json`, CSV tables, and a zip bundle."
                )
                manifest = gr.Dataframe(
                    headers=["artifact", "purpose", "path"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    label="Artifact manifest",
                )
                downloads = gr.File(
                    label="Download files",
                    file_count="multiple",
                    type="filepath",
                )

        analyze_btn.click(
            fn=_run_analysis,
            inputs=[upload, sample_state],
            outputs=[status, summary, summary_table, findings, report_markdown, downloads, manifest],
        )
        if sample_path:
            sample_btn.click(
                fn=lambda sp: _run_analysis(None, sp),
                inputs=[sample_state],
                outputs=[status, summary, summary_table, findings, report_markdown, downloads, manifest],
            )
            demo.load(
                fn=lambda sp: _run_analysis(None, sp),
                inputs=[sample_state],
                outputs=[status, summary, summary_table, findings, report_markdown, downloads, manifest],
            )
    return demo


def main() -> None:
    demo = _build_demo(find_local_sample())
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
