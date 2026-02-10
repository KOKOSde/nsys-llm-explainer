## nsys-llm-explainer

[![CI](https://github.com/KOKOSde/nsys-llm-explainer/actions/workflows/ci.yml/badge.svg)](https://github.com/KOKOSde/nsys-llm-explainer/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

![nsys-llm-explainer hero diagram](docs/hero.svg)

### Why this exists

Nsight Systems traces are powerful, but the SQLite export is still hard to interpret when you’re chasing LLM inference bottlenecks.
This tool turns a `trace.sqlite` into a concise, actionable report: top kernels, launch storms, sync indicators, GPU idle gaps, NVTX phases, and per-PID breakdowns.
It is designed for **vLLM-style multi-process traces** (server + workers) and is **A100-first** (capture/runbook assumes A100).
Every number in the report is derived from concrete SQLite queries, and the report includes explicit limitations/coverage so you can judge confidence.

### Design principles

- **Offline-only**: consumes the exported SQLite (`trace.sqlite`), no live profiling or network access required.
- **Trace-derived metrics**: report numbers come from specific SQLite tables/columns (and the report calls out what is missing when schema differs).
- **Explicit uncertainty**: NVTX→kernel attribution is best-effort and reports coverage; PID attribution is best-effort and warns when it looks suspicious.
- **No benchmark claims**: this is not a microbenchmark suite; it does not claim speedups.

### Install (editable)

```bash
python3.9 -m pip install -e .
```

### Minimal reproducible commands

#### 1) Capture a small trace with Nsight Systems

Minimal capture (CUDA + NVTX + OS runtime):

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  -o trace \
  python your_workload.py
```

Notes:

- `--cuda-graph-trace=node` matters for workloads that use CUDA graphs (common in inference stacks).
- Keep the capture short (seconds, not minutes) and focused on steady-state inference.

#### 2) Export SQLite

```bash
nsys export --type sqlite --output trace.sqlite --force-overwrite=true --lazy=false trace.nsys-rep
```

#### 3) Run the explainer

```bash
nsys-llm-explain trace.sqlite --out artifacts/run_YYYYMMDD_HHMMSS/
```

Optional NVTX phase mapping (maps arbitrary NVTX range names into phases like `prefill` / `decode` / `sampling`):

```bash
nsys-llm-explain trace.sqlite --out artifacts/run_YYYYMMDD_HHMMSS/ --phase-map phases.json
```

`--phases-json` is accepted as an alias for `--phase-map`.

You can tune when the report emits a warning for low NVTX→kernel attribution coverage:

```bash
nsys-llm-explain trace.sqlite --out artifacts/run_YYYYMMDD_HHMMSS/ --nvtx-coverage-warn-threshold 0.7
```

Example `phases.json`:

```json
{
  "prefill": ["prefill", "prompt"],
  "decode": ["decode", "generation"],
  "sampling": ["sample", "sampling"]
}
```

### Capture steady-state guidance (to avoid misleading “idle gap %”)

The report’s “GPU idle” is computed over the **observed kernel time window** in the SQLite export. If your capture includes:

- model load / compilation
- a long warmup before decode
- a long idle tail after requests finish

…then the window can be dominated by periods with **no kernels**, and “idle %” can look extreme even if steady-state decode is fine.

Recommended practice:

- **Warm up first**, then start the capture.
- Capture only a short, steady-state window around decode (seconds).
- If you use NVTX, consider capturing only a specific NVTX range (Nsight Systems supports capture-range options; see `nsys profile --help` for your version).

### Generate `trace.sqlite` (from an existing `.nsys-rep`)

Capture a trace and export SQLite:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -o trace python your_workload.py
nsys export --type sqlite --output trace.sqlite your_trace.nsys-rep
```

### Outputs

The output directory will contain:

- `report.md`: PR-ready Markdown report
- `report.json`: machine-readable metrics + recommendations
- `tables/kernels.csv`: top kernels table
- `tables/nvtx_ranges.csv`: NVTX ranges (if present)
- `tables/gpu_idle_gaps.csv`: estimated idle gaps (if computed)
- `tables/kernels_by_pid.csv`: top kernels grouped by PID (best-effort, if PID info is available)
- `tables/sync_by_pid.csv`: sync-like runtime calls grouped by PID (best-effort)
- `tables/nvtx_by_pid.csv`: NVTX ranges grouped by PID (best-effort; written when NVTX is present)

### Example: real A100 vLLM trace

See `examples/a100_vllm/` for a committed, real capture (outputs only). The raw `trace.sqlite` is intentionally omitted to keep the repo small.

Report excerpt (from `examples/a100_vllm/report.md`):

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

The generated `report.md` includes:

- A “What to do next” section with trace-backed findings
- Tables for top kernels, sync-like API calls, idle gaps, and NVTX (if present)

### What the report measures (trace-derived)

- **Top CUDA kernels**: from `CUPTI_ACTIVITY_KIND_KERNEL` (GPU kernel intervals), names resolved via `StringIds`.
- **Launch storm**: kernel launches/sec + duration percentiles derived from kernel intervals.
- **CPU↔GPU sync indicators**: runtime/driver API call durations from `CUPTI_ACTIVITY_KIND_RUNTIME` filtered to sync-like calls (e.g., `cudaDeviceSynchronize`, `cudaStreamSynchronize`, waits).
- **GPU idle gaps (estimate)**: per-device union of kernel intervals to estimate busy vs idle within the kernel time window.
- **NVTX breakdown**: `NVTX_EVENTS` rows with `end` timestamps summarized by range name; optional mapping of range names into phases via `--phase-map`.
- **NVTX-attributed GPU kernel time (best-effort)**: if `correlationId` + `globalTid` are present, attributes GPU kernel time to enclosing NVTX ranges via NVTX→runtime→kernel correlation; can be disabled with `--no-nvtx-kernel-map`.
- **Multi-process (PID) breakdown (best-effort)**: top PIDs by kernel time, per-PID top kernels, and per-PID sync-like calls when PID columns are available in the export.

### What it does not do

- It does **not** claim benchmark performance or “X% speedups”. It reports what the trace shows.
- It does **not** attempt a perfect GPU utilization model across overlapping streams; “idle/busy” is a conservative estimate from kernel intervals.
- It does **not** guarantee NVTX phase attribution is complete; kernel-time attribution is best-effort and includes coverage reporting.

### What good looks like (heuristics)

These are intentionally simple, trace-based heuristics to help triage. They are not proofs.

- **Kernel launch storm**:
  - Report classification uses thresholds from `src/nsys_llm_explainer/heuristics.py`:
    - launch storm if \(launches/s \ge 50{,}000\) and \(p50 \le 10\,\mu s\), or \(launches/s \ge 100{,}000\) and \(p50 \le 20\,\mu s\).
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

- **String table**: prefers `StringIds(id,value)`, falls back to any table with `id` + `value` for name resolution.
- **Kernel activity**: prefers `CUPTI_ACTIVITY_KIND_KERNEL`, falls back to `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` if present.
- **Runtime API**: prefers `CUPTI_ACTIVITY_KIND_RUNTIME` (for sync indicators and correlationId mapping).
- **NVTX**: prefers `NVTX_EVENTS` (for wall-time ranges; kernel-time attribution requires correlationId/globalTid availability).
- **Per-PID breakdowns**: best-effort decoding from available PID-bearing columns (`pid`, `processId`, `globalPid`, `globalTid`). The report warns when PID decoding looks suspicious for an export.

### Limitations / schema differences

- Nsight Systems tables are created lazily; not all tables are present in every export. The tool probes schema at runtime and degrades gracefully.
- Timestamps are interpreted as **nanoseconds** (Nsight Systems CUPTI exports) and converted to ms/us. If an export uses a different unit scale, time-derived values will be wrong; the report warns when it cannot run a sanity check.
- Idle/busy is **kernel-interval based** (does not include non-kernel GPU work unless you extend it to include memcpy/memset workloads).
- NVTX phase attribution depends on NVTX being present, and on Nsight exporting `correlationId`/`globalTid` needed to correlate kernels back to NVTX ranges. Coverage may be partial.
- Per-PID sections depend on PID-bearing columns (`globalPid` / `globalTid` / `pid` / `processId`). The report will emit a warning if PID attribution looks ambiguous.

### Schema fallbacks (what the tool probes)

- **String table**: prefers `StringIds(id,value)`, falls back to any table with `id` + `value`.
- **Kernel activity**: prefers `CUPTI_ACTIVITY_KIND_KERNEL`, falls back to `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` if present.
- **Runtime API**: prefers `CUPTI_ACTIVITY_KIND_RUNTIME`.
- **NVTX**: prefers `NVTX_EVENTS`.

### References

- Nsight Systems SQLite exporter schema reference: `https://docs.nvidia.com/nsight-systems/` (see `nsys-exporter` docs for your version).

