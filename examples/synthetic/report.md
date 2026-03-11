# Nsight Systems LLM Hotspot Report

> Synthetic example generated from `tests/test_synthetic_sqlite.py::_build_trace_with_nccl_and_barriers`.

- Generated at (UTC): `2026-03-11T03:52:08.704882+00:00`
- Trace: `synthetic fixture (raw trace.sqlite not committed)`
- Tool: `nsys-llm-explain 0.1.0`

## Warnings

- NVLink counters not found. The report cannot correlate NCCL windows with NVLink metrics for this export.

## What to do next

- **[medium] Single kernel is a large share of GPU time**
  - **Evidence**:
    - Top kernel `computeKernel` is 42.6% of total kernel time.
  - **Recommendation**:
    - Focus optimization effort on this kernel first.
- **[medium] CPU↔GPU synchronization detected (runtime API)**
  - **Evidence**:
    - Top sync-like call `cudaStreamSynchronize` total 0.80 ms across 1 calls.
    - All sync-like calls total 1.50 ms.
  - **Recommendation**:
    - Look for `cudaDeviceSynchronize` / stream waits in your serving loop and remove unnecessary barriers.
    - Prefer async launches and overlap CPU work with GPU execution; avoid per-token synchronization.

## Global critical path suspects

| kind | name | total_ms | count | details |
| --- | --- | --- | --- | --- |
| kernel | computeKernel | 2.600 | 3 | 42.6% of kernel time |
| nccl | allreduce | 2.000 | 1 | max 2.000 ms |
| gpu_idle | GPU 0 | 1.000 | 1 | 18.2% idle |
| barrier | cudaStreamSynchronize | 0.800 | 1 | sync_api |

## Top NCCL ops

- **Derived from**: best-effort NCCL windows from NVTX ranges, runtime API calls, or NCCL kernel names.
- **Limitations**: op names are only as precise as the exported trace data; kernel-only traces may yield raw NCCL kernel names instead of collective labels.

| op_name | source | total_time_ms | max_duration_ms | count | compute_overlap_ms | compute_overlap_pct | straggler |
| --- | --- | --- | --- | --- | --- | --- | --- |
| allreduce | kernel | 2.000 | 2.000 | 1 | 1.000 | 50.0 | pid:111 |
| broadcast | kernel | 1.500 | 1.500 | 1 | 0.600 | 40.0 | pid:222 |

- Using NCCL kernel names as NCCL windows; collective names may be inferred only from kernel names.

## NVLink during NCCL

- **Derived from**: `GPU_METRICS` / `TARGET_INFO_GPU_METRICS` samples aligned with NCCL-active windows.
- **Limitations**: GPU Metrics are device-level samples; they are not process-attributed in the SQLite export.

- GPU metric tables were not found in this export.
- NVLink counters not found in the SQLite export.
- List supported metric sets first: `nsys profile --gpu-metrics-devices=all --gpu-metrics-set=help`.
- Then re-capture with GPU Metrics enabled, for example: `sudo nsys profile --trace=nccl,cuda,nvtx,osrt --cuda-trace-scope=process-tree --gpu-metrics-devices=all --gpu-metrics-set=<supported-set> --gpu-metrics-frequency=10000 --cuda-graph-trace=node -o trace <app>`.
- Export again with SQLite output: `nsys export --type sqlite --output trace.sqlite --force-overwrite=true --lazy=false trace.nsys-rep`.

## Global: top CUDA kernels (by total time)

- **Derived from**: `CUPTI_ACTIVITY_KIND_KERNEL`; duration = `end-start`.
- **Limitations**: totals are summed over launches (no overlap correction); names may be numeric IDs if string resolution is unavailable.

| kernel_name | device_id | total_ms | calls | avg_us | p50_us | p90_us | pct_kernel_time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| computeKernel | 0 | 2.600 | 3 | 866.67 | 1000.00 | 1000.00 | 42.6 |
| ncclAllReduceRingKernel | 0 | 2.000 | 1 | 2000.00 | 2000.00 | 2000.00 | 32.8 |
| ncclBroadcastRingKernel | 0 | 1.500 | 1 | 1500.00 | 1500.00 | 1500.00 | 24.6 |

## Per-process breakdown

- **Derived from**: per-PID kernel, NCCL, and barrier aggregations when PID-bearing columns exist.
- **Limitations**: PID attribution is best-effort and depends on exported `pid`/`processId`/`globalPid`/`globalTid` columns.

| pid | kernel_time_ms | kernel_count | nccl_time_ms | barrier_time_ms | top_nccl_op | top_barrier |
| --- | --- | --- | --- | --- | --- | --- |
| 111 | 4.000 | 3 | 2.000 | 1.600 | allreduce | cudaStreamSynchronize |
| 222 | 2.100 | 2 | 1.500 | 0.700 | broadcast | cudaDeviceSynchronize |

### Top PIDs by GPU kernel time

- **Derived from**: `CUPTI_ACTIVITY_KIND_KERNEL` grouped by PID (requires kernel PID column such as `globalPid`).
- **PID source**: `globalPid`

| pid | total_kernel_time_ms | kernel_count | pct_of_total_kernel_time |
| --- | --- | --- | --- |
| 111 | 4.000 | 3 | 65.6 |
| 222 | 2.100 | 2 | 34.4 |

### Top kernels per PID

### PID `111`

- PID kernel time: `4.000 ms`

| kernel_name | device_id | total_time_ms | call_count | avg_duration_us | pct_of_pid_kernel_time |
| --- | --- | --- | --- | --- | --- |
| computeKernel | 0 | 2.000 | 2 | 1000.00 | 50.0 |
| ncclAllReduceRingKernel | 0 | 2.000 | 1 | 2000.00 | 50.0 |

### PID `222`

- PID kernel time: `2.100 ms`

| kernel_name | device_id | total_time_ms | call_count | avg_duration_us | pct_of_pid_kernel_time |
| --- | --- | --- | --- | --- | --- |
| ncclBroadcastRingKernel | 0 | 1.500 | 1 | 1500.00 | 71.4 |
| computeKernel | 0 | 0.600 | 1 | 600.00 | 28.6 |

## Top CPU↔GPU barriers

- **Derived from**: CUDA runtime calls, blocking memcpy APIs, host wait APIs (when `OSRT_API` exists), and synthetic CPU launcher gaps between launch APIs.
- **Limitations**: OS runtime waits are only available when the export contains OSRT tables; launcher gaps are host-side gaps between launch calls, not direct GPU counters.

| barrier_kind | api_name | total_time_ms | count | avg_duration_us | max_duration_us |
| --- | --- | --- | --- | --- | --- |
| sync_api | cudaStreamSynchronize | 0.800 | 1 | 800.00 | 800.00 |
| sync_api | cudaDeviceSynchronize | 0.700 | 1 | 700.00 | 700.00 |
| blocking_memcpy | cudaMemcpy | 0.600 | 1 | 600.00 | 600.00 |
| cpu_launcher_gap | cpu_launcher_gap | 0.200 | 1 | 200.00 | 200.00 |

## Launch storm per PID (best-effort)

- **Derived from**: `CUPTI_ACTIVITY_KIND_KERNEL` kernel timestamps filtered by PID.
- **Limitations**: per-PID launch storm depends on PID decoding; overlap across streams does not invalidate launch rate but complicates interpretation.

| pid | total_launches | window_s | launches_per_s | p50_kernel_us | p90_kernel_us | p99_kernel_us | pct_under_5us | pct_under_10us | pct_under_20us | launch_storm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 222 | 2 | 0.001500 | 1333.333 | 600.00 | 1500.00 | 1500.00 | 0.0 | 0.0 | 0.0 | false |
| 111 | 3 | 0.003000 | 1000.000 | 1000.00 | 2000.00 | 2000.00 | 0.0 | 0.0 | 0.0 | false |

## Sync indicators per PID

- **Derived from**: `CUPTI_ACTIVITY_KIND_RUNTIME` runtime API intervals grouped by PID (requires runtime `globalTid`/pid).
- **Limitations**: only reports what was traced/exported; some waits may not appear as explicit sync calls.
- **PID source**: `globalTid`

| pid | api_name | total_time_ms | call_count | avg_duration_us |
| --- | --- | --- | --- | --- |
| 111 | cudaStreamSynchronize | 0.800 | 1 | 800.00 |
| 222 | cudaDeviceSynchronize | 0.700 | 1 | 700.00 |

## Global: launch storm

- **Derived from**: `CUPTI_ACTIVITY_KIND_KERNEL` kernel timestamps (`start/end`).
- **Limitations**: uses kernel-table window; overlap across streams doesn’t invalidate launch rate but complicates “GPU saturated” interpretation.

- launches: `5` over `0.005s` = `909.1/s`
- duration p50/p90/p99 (us): `1000.00` / `1800.00` / `1980.00`
- % kernels under 5/10/20 us: `0.0%` / `0.0%` / `0.0%`
- launch_storm = `False`

## Global: CPU↔GPU synchronization (CUDA runtime/driver)

- **Derived from**: `CUPTI_ACTIVITY_KIND_RUNTIME` API intervals filtered by sync-like names.
- **Limitations**: only reports what was traced/exported; some waits may not appear as explicit sync calls.

| api_name | call_count | total_time_ms | avg_duration_us |
| --- | --- | --- | --- |
| cudaStreamSynchronize | 1 | 0.800 | 800.00 |
| cudaDeviceSynchronize | 1 | 0.700 | 700.00 |

## GPU idle estimate (from kernel timeline)

- **Derived from**: union of kernel intervals from `CUPTI_ACTIVITY_KIND_KERNEL` (per device if `deviceId` exists).
- **Limitations**: approximate/conservative; excludes memcpy/memset/other GPU activities; overlap across streams is merged (union).

| device_id | window_ms | busy_ms | idle_ms | idle_pct_of_window |
| --- | --- | --- | --- | --- |
| 0 | 5.500 | 4.500 | 1.000 | 18.2 |

Largest gaps:

| device_id | gap_start_ns | gap_end_ns | gap_ms |
| --- | --- | --- | --- |
| 0 | 3000000 | 4000000 | 1.000 |

## NVTX ranges

- **Derived from**: `None` rows with non-null `end`, aggregated by range name.
- **Limitations**: NVTX is host-side timing; it does not directly measure GPU time without additional correlation.

_(no NVTX ranges found)_

## NVTX per PID (best-effort)

- **Derived from**: `None` grouped by PID (requires NVTX `globalTid`/pid).
- **Limitations**: depends on exported NVTX columns; host-side only; GPU attribution is best-effort if present elsewhere in report.
- **PID source**: `None`

_(no NVTX PID breakdown available)_

## Derivation & assumptions

- **Timestamp units**: report interprets `start/end` as **nanoseconds** and converts to ms/us via `/1e6` and `/1e3`.
- **Timestamp sanity check**: `timestamp_unit_guess=ns_likely` (basis `kernel_window_ns_ge_1ms`). If `unknown`, treat time-derived numbers as suspect.
- **Kernel durations**: `end-start` from `CUPTI_ACTIVITY_KIND_KERNEL` summed over launches (no overlap correction).
- **NCCL windows**: best-effort from NVTX ranges first, then runtime API names, then NCCL kernel names. Collective labels may degrade to raw names when only kernels are available.
- **NVLink during NCCL**: computed only when GPU Metrics tables (`GPU_METRICS`, `TARGET_INFO_GPU_METRICS`) include NVLink-related metrics; otherwise the report emits capture instructions instead of guessing.
- **CPU launcher gaps**: host-side gaps between consecutive CUDA launch APIs above the configured threshold; this is a barrier heuristic, not a hardware counter.
- **GPU idle estimate**: per-device union of kernel intervals within the kernel time window; excludes memcpy/memset unless you extend the tool.
- **NVTX→kernel attribution**: best-effort correlation (`kernel.correlationId` → runtime launch site → `globalTid` → enclosing NVTX range). Coverage is reported; low coverage means per-phase attribution may not reflect total GPU time.
- **Per-PID attribution**: best-effort decoding from available PID-bearing columns (`pid`, `processId`, `globalPid`, `globalTid`).
- **CUDA graphs capture**: if your workload uses CUDA Graphs, capture with `--cuda-graph-trace=node` so graph launches remain visible in the export.
