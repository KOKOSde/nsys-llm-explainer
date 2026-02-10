## nsys-llm-explainer

[![CI](https://github.com/KOKOSde/nsys-llm-explainer/actions/workflows/ci.yml/badge.svg)](https://github.com/KOKOSde/nsys-llm-explainer/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

![nsys-llm-explainer hero diagram](docs/hero.svg)

### Why this exists

Nsight Systems traces are powerful, but the SQLite export is still hard to interpret when you’re chasing LLM inference bottlenecks.
This tool turns a `trace.sqlite` into a concise report: top kernels, launch storms, sync indicators, GPU idle gaps, NVTX ranges, and per-PID breakdowns.
It is designed for **vLLM-style multi-process traces** and is **A100-first** (capture/runbook assumes A100).
Every number is trace-derived and the report calls out coverage/limitations explicitly.

### Install (editable)

```bash
python3.9 -m pip install -e .
```

### Quickstart (minimal commands)

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  -o trace \
  python your_workload.py

nsys export --type sqlite --output trace.sqlite --force-overwrite=true --lazy=false trace.nsys-rep
nsys-llm-explain trace.sqlite --out artifacts/run_YYYYMMDD_HHMMSS/
```

Notes:

- `--cuda-graph-trace=node` matters for workloads that use CUDA graphs.
- Optional NVTX phase mapping: `--phase-map phases.json` (alias: `--phases-json`).

### Outputs

The output directory contains:

- `report.md`, `report.json`
- `tables/kernels.csv`
- `tables/gpu_idle_gaps.csv` (if computed)
- `tables/nvtx_ranges.csv` (if present)
- `tables/kernels_by_pid.csv`, `tables/sync_by_pid.csv`, `tables/nvtx_by_pid.csv` (best-effort, if PID/NVTX info exists)

### Example: real A100 vLLM trace

See `examples/a100_vllm/` for a committed, real capture (outputs only). The raw `trace.sqlite` is intentionally omitted to keep the repo small.

Excerpt (from `examples/a100_vllm/report.md`):

```text
## Warnings

- NVTX-attributed GPU time is best-effort (NVTX→runtime→kernel correlation). Coverage is 2.2% (< 70.0%). Low coverage → interpret cautiously.
- Per-PID NVTX-attributed GPU time has low coverage for at least one PID (worst PID 495200: 2.2%). Interpret per-phase/per-PID attribution cautiously.

## What to do next

- **[medium] CPU↔GPU synchronization detected (runtime API)**
  - **Evidence**:
    - Top sync-like call `cudaEventSynchronize_v3020` total 129.97 ms across 129 calls.
    - All sync-like calls total 233.63 ms.
- **[high] Significant GPU idle gaps**
  - **Evidence**:
    - GPU 0 idle 99.4% of observed window (82727.3 ms / 83203.8 ms).

## Global: top CUDA kernels (by total time)
| kernel_name | device_id | total_ms | calls | avg_us | p50_us | p90_us | pct_kernel_time |
| ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn | 0 | 47.015 | 3612 | 13.02 | 12.29 | 16.86 | 9.9 |
```

### Steady-state capture guidance (idle% can be misleading)

“GPU idle” is computed over the **observed kernel time window** (first kernel start → last kernel end). If your capture includes model load, long warmup, or a long idle tail, idle% can look extreme.

Recommended practice:

- Warm up first, then start the capture.
- Capture a short steady-state window (seconds).
- If your Nsight Systems version supports capture-range options, consider restricting capture to an NVTX range (see `nsys profile --help`).

### What the report measures (trace-derived)

- **Top CUDA kernels**: from `CUPTI_ACTIVITY_KIND_KERNEL` (GPU kernel intervals), names resolved via `StringIds`.
- **Launch storm**: kernel launches/sec + duration percentiles derived from kernel intervals.
- **CPU↔GPU sync indicators**: runtime/driver API call durations from `CUPTI_ACTIVITY_KIND_RUNTIME` filtered to sync-like calls (e.g., `cudaDeviceSynchronize`, `cudaStreamSynchronize`, waits).
- **GPU idle gaps (estimate)**: per-device union of kernel intervals to estimate busy vs idle within the kernel time window.
- **NVTX breakdown**: `NVTX_EVENTS` rows with `end` timestamps summarized by range name; optional mapping of range names into phases via `--phase-map`.
- **NVTX-attributed GPU kernel time (best-effort)**: if `correlationId` + `globalTid` are present, attributes GPU kernel time to enclosing NVTX ranges via NVTX→runtime→kernel correlation; can be disabled with `--no-nvtx-kernel-map`.
- **Multi-process (PID) breakdown (best-effort)**: top PIDs by kernel time, per-PID top kernels, and per-PID sync-like calls when PID columns are available in the export.

### Design principles / non-goals

- Offline-only, trace-derived metrics only.
- Reports coverage/limitations instead of implying certainty.
- No benchmark claims or speedup promises.

### What good looks like (heuristics)

- **Kernel launch storm**: classified using thresholds in `src/nsys_llm_explainer/heuristics.py` (high launches/sec + tiny median kernel).
- **Dominant kernel**:
  - If the top kernel is **≥ 50%** of total kernel time, that is usually the first place to focus (single hotspot dominates).
- **Sync calls**:
  - Frequent `cudaDeviceSynchronize` / `cudaStreamSynchronize` / `cudaEventSynchronize` can indicate CPU↔GPU barriers that reduce overlap.
- **CPU-bound signatures**:
  - Large GPU idle gaps + many short kernels can be consistent with CPU scheduling/launch overhead or unnecessary synchronization.
- **NVTX phase interpretation**:
  - NVTX wall-time is host timing. NVTX-attributed GPU kernel time (if present) is best-effort correlation and must be interpreted with its reported coverage.

### Schema compatibility

- **Tested export**: the committed example was captured/exported in the ASU environment; see `examples/a100_vllm/metadata.txt` for the recorded `nsys --version` output.
- **Graceful degradation**: the tool probes the SQLite schema at runtime and only emits sections it can compute from available tables/columns.

Key probes and fallbacks:

- **String table**: prefers `StringIds(id,value)`, falls back to any `id`+`value` table.
- **Kernel activity**: prefers `CUPTI_ACTIVITY_KIND_KERNEL`, falls back to `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`.
- **Runtime API**: prefers `CUPTI_ACTIVITY_KIND_RUNTIME`.
- **NVTX**: prefers `NVTX_EVENTS`.

### Limitations / schema differences

- Nsight Systems tables are created lazily; not all tables are present in every export. The tool probes schema at runtime and degrades gracefully.
- Timestamps are interpreted as **nanoseconds** (Nsight Systems CUPTI exports) and converted to ms/us. If an export uses a different unit scale, time-derived values will be wrong; the report warns when it cannot run a sanity check.
- Idle/busy is **kernel-interval based** (does not include non-kernel GPU work unless you extend it to include memcpy/memset workloads).
- NVTX phase attribution depends on NVTX being present, and on Nsight exporting `correlationId`/`globalTid` needed to correlate kernels back to NVTX ranges. Coverage may be partial.
- Per-PID sections depend on PID-bearing columns (`globalPid` / `globalTid` / `pid` / `processId`). The report will emit a warning if PID attribution looks ambiguous.

### References

- Nsight Systems SQLite exporter schema reference: `https://docs.nvidia.com/nsight-systems/` (see `nsys-exporter` docs for your version).

