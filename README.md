## nsys-llm-explainer

[![CI](https://github.com/KOKOSde/nsys-llm-explainer/actions/workflows/ci.yml/badge.svg)](https://github.com/KOKOSde/nsys-llm-explainer/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

![nsys-llm-explainer hero diagram](docs/hero.svg)

### Why this exists

Nsight Systems traces are powerful, but the SQLite export is still hard to interpret when you’re chasing LLM inference bottlenecks.
This tool turns a `trace.sqlite` into a concise, actionable report: top kernels, launch storms, sync indicators, GPU idle gaps, NVTX phases, and per-PID breakdowns.
It is designed for **vLLM-style multi-process traces** (server + workers) and is **A100-first** (capture/runbook assumes A100).
Every number in the report is derived from concrete SQLite queries, and the report includes explicit limitations/coverage so you can judge confidence.

### Install (editable)

```bash
python3.9 -m pip install -e .
```

### Generate `trace.sqlite`

Capture a trace and export SQLite:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -o trace python your_workload.py
nsys export --type sqlite --output trace.sqlite your_trace.nsys-rep
```

### Run

```bash
nsys-llm-explain trace.sqlite --out artifacts/run_YYYYMMDD_HHMMSS/
```

Optional NVTX phase mapping (maps arbitrary NVTX range names into phases like `prefill`/`decode`/`sampling`):

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

## What to do next

- [medium] CPU↔GPU synchronization detected (runtime API)
  - Evidence: Top sync-like call cudaEventSynchronize total 129.97 ms across 129 calls.
- [high] Significant GPU idle gaps
  - Evidence: GPU 0 idle 99.4% of observed window (82727.3 ms / 83203.8 ms).

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

