# Hugging Face Cloud ML Engineer Resume Bullets

Use these as resume bullets or talking points. Replace placeholders like `<X%>` and `<N>` with your real numbers.

## Positioning

- Cloud ML engineer who builds production-grade inference and profiling tooling across GPUs, containers, dashboards, and documentation.
- Strong fit for Hugging Face ML Cloud because the work sits at the intersection of cloud deployment, developer experience, performance analysis, and open-source reuse.

## Resume Bullets

- Built `nsys-llm-explainer`, an offline Nsight Systems SQLite analyzer for LLM inference that turns raw traces into prioritized, evidence-backed findings across kernels, barriers, NCCL, NVLink, launch latency, stream overlap, and per-process breakdowns.
- Designed a reproducible trace-analysis pipeline that emits both `report.md` and structured JSON/CSV artifacts, enabling fast debugging, automated regression checks, and downstream dashboard integration across `<N>` traces and `<M>` GPU profiles.
- Implemented conservative correlation logic for NCCL and NVLink that only reports relationships when the export contains valid metrics, and surfaces missing-counter warnings instead of fabricating results.
- Added dashboard workflows for current-vs-baseline trace comparison in a dark-theme Plotly/Dash app, improving performance regression triage time by `<X%>` across repeated runs and making bottlenecks visible in one screen.
- Built a report system that detects launch storms, CPU-GPU barriers, idle gaps, and top-kernel hotspots, helping engineers prioritize fixes that improve GPU utilization and reduce end-to-end latency.
- Packaged the tool as a clean Python project with CLI entrypoints, reproducible artifact generation, and test coverage (`9/9` synthetic analysis tests passing in the current suite).
- Created a clear path to a Hugging Face Space deployment by separating the analysis engine from the UI, which makes the project easy to wrap in Gradio or expose through a small cloud service.
- Created documentation that explains how to capture and export Nsight Systems traces correctly for CUDA, NCCL, NVTX, and CUDA Graphs workloads, reducing first-run setup friction for new users.
- Produced a dashboard-ready output format with top events, copy-engine activity, launch-latency distributions, stream overlap summaries, phase splits, and roofline metrics for cloud inference tuning.
- Shipped a dark-theme interactive dashboard that accepts `.sqlite` and `.json` inputs, supports baseline comparison, and provides one-click static export for sharing with teammates or partner teams.
- Operated with a guardrail-first approach: explicit limitations, warnings for missing counters, and best-effort PID attribution so decisions are defensible in production environments.

## Additions To Make It Stronger

- Publish a live Hugging Face Space titled `nsys-llm-explainer — Instant Nsight Trace Analyzer for Cloud LLM Inference` that demonstrates the analyzer on sample traces and shows the cloud-ready workflow end to end.
- Add one cloud deployment example for the analyzer service itself, ideally containerized and deployable to a managed platform in `<N>` minutes, with a README section that maps cleanly to Hugging Face Spaces deployment.
- Document one real performance win from a trace you analyzed, with a before/after chart and the exact metric improvement.
