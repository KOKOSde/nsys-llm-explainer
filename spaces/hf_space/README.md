---
title: nsys-llm-explainer — Instant Nsight Trace Analyzer for Cloud LLM Inference
emoji: "📈"
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---

# nsys-llm-explainer — Instant Nsight Trace Analyzer for Cloud LLM Inference

This folder is a production-ready Hugging Face Space payload for the `nsys-llm-explainer` project.

It turns an uploaded `trace.sqlite`, `.db`, or `report.json` into:

- Prioritized findings with evidence and recommendations
- Kernel, NCCL, barrier, and launch-latency summaries
- NVLink-over-NCCL correlation when GPU metrics are available
- Markdown preview of the full report
- Downloadable `report.md`, `report.json`, CSV tables, and a zip bundle

## Files

- `app.py`: Gradio app entrypoint
- `space_utils.py`: analysis and artifact helpers
- `requirements.txt`: Space dependencies

## Deploy on Hugging Face Spaces

1. Create a new Space using the `Gradio` SDK.
2. Copy the contents of this folder into the Space repository root.
3. Keep `requirements.txt` in place so the Space installs the analyzer package and Gradio runtime.
4. Push the repo. Hugging Face will build the Space automatically.
5. Open the app and upload a `trace.sqlite` or `report.json`.

## Duplicate and pin

If you want a reproducible Space, keep the Git dependency pinned to a release tag in `requirements.txt`.

If you want the Space to follow the latest `main` branch instead, change:

```txt
git+https://github.com/KOKOSde/nsys-llm-explainer.git@v0.3.2
```

to:

```txt
git+https://github.com/KOKOSde/nsys-llm-explainer.git@main
```

## Operational notes

- The app works with uploaded SQLite exports directly, so there is no need to pre-generate artifacts.
- If a trace is missing NCCL or GPU metrics tables, the UI still loads and explains which analyses are unavailable.
- For private traces, use a private Space.

## Local run

From this repository root:

```bash
PYTHONPATH=src python3 spaces/hf_space/app.py
```

If you are running the folder standalone, first install the dependencies from `requirements.txt`.
