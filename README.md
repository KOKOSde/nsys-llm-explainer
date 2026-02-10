## nsys-llm-explainer

Offline tool that turns an Nsight Systems **SQLite export** into an actionable performance report for **LLM inference** (vLLM-focused, but generally useful).

### Install (editable)

```bash
python3.9 -m pip install -e .
```

### Generate `trace.sqlite`

From a collected `.nsys-rep`:

```bash
nsys export --type sqlite --output trace.sqlite your_trace.nsys-rep
```

### Run

```bash
nsys-llm-explain trace.sqlite --out artifacts/<run_id>/
```

Optional NVTX phase mapping (maps arbitrary NVTX range names into phases like `prefill`/`decode`/`sampling`):

```bash
nsys-llm-explain trace.sqlite --out artifacts/<run_id>/ --phase-map phases.json
```

`--phases-json` is accepted as an alias for `--phase-map`.

You can tune when the report emits a warning for low NVTX→kernel attribution coverage:

```bash
nsys-llm-explain trace.sqlite --out artifacts/<run_id>/ --nvtx-coverage-warn-threshold 0.7
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

### Example report excerpt (shape)

The generated `report.md` includes:

- A “What to do next” section with trace-backed findings
- Tables for top kernels, sync-like API calls, idle gaps, and NVTX (if present)

Example (illustrative):

```text
## Top CUDA kernels (by total time)
| kernel_name | device_id | total_ms | calls | avg_us | p50_us | p90_us | pct_kernel_time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flash_fwd | 0 | 123.456 | 2048 | 60.27 | 58.10 | 75.33 | 42.1 |
```

### What the report measures (trace-derived)

- **Top CUDA kernels**: from `CUPTI_ACTIVITY_KIND_KERNEL` (GPU kernel intervals), names resolved via `StringIds`.
- **Launch storm**: kernel launches/sec + duration percentiles derived from kernel intervals.
- **CPU↔GPU sync indicators**: runtime/driver API call durations from `CUPTI_ACTIVITY_KIND_RUNTIME` filtered to sync-like calls (e.g., `cudaDeviceSynchronize`, `cudaStreamSynchronize`, waits).
- **GPU idle gaps (estimate)**: per-device union of kernel intervals to estimate busy vs idle within the kernel time window.
- **NVTX breakdown**: `NVTX_EVENTS` rows with `end` timestamps summarized by range name; optional mapping of range names into phases via `--phase-map`.
- **NVTX-attributed GPU kernel time (best-effort)**: if `correlationId` + `globalTid` are present, attributes GPU kernel time to enclosing NVTX ranges via NVTX→runtime→kernel correlation; can be disabled with `--no-nvtx-kernel-map`.
- **Multi-process (PID) breakdown (best-effort)**: top PIDs by kernel time, per-PID top kernels, and per-PID sync-like calls when PID columns are available in the export.

### Limitations / schema differences

- Nsight Systems tables are created lazily; not all tables are present in every export. The tool probes schema at runtime and degrades gracefully.
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

