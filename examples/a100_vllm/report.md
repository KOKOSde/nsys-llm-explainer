# Nsight Systems LLM Hotspot Report

- Generated at (UTC): `2026-02-10T08:18:55.746213+00:00`
- Trace: `artifacts/example_a100_vllm_20260210T081135Z/trace.sqlite`
- Tool: `nsys-llm-explain 0.1.0`

## Warnings

- NVTX-attributed GPU time is best-effort (NVTX→runtime→kernel correlation). Coverage is 2.2% (< 70.0%). Low coverage → interpret cautiously.
- Per-PID NVTX-attributed GPU time has low coverage for at least one PID (worst PID 495200: 2.2%). Interpret per-phase/per-PID attribution cautiously.

## What to do next

- **[medium] CPU↔GPU synchronization detected (runtime API)**
  - **Evidence**:
    - Top sync-like call `cudaEventSynchronize_v3020` total 129.97 ms across 129 calls.
    - All sync-like calls total 233.63 ms.
  - **Recommendation**:
    - Look for `cudaDeviceSynchronize` / stream waits in your serving loop and remove unnecessary barriers.
    - Prefer async launches and overlap CPU work with GPU execution; avoid per-token synchronization.
- **[high] Significant GPU idle gaps**
  - **Evidence**:
    - GPU 0 idle 99.4% of observed window (82727.3 ms / 83203.8 ms).
  - **Recommendation**:
    - If kernels are short, CPU scheduling/launch overhead can create gaps; batch more work per launch.
    - Check for sync calls and data dependencies that serialize work; increase overlap across streams.
- **[low] NVTX ranges present**
  - **Evidence**:
    - Found 3 NVTX range names.
  - **Recommendation**:
    - Use NVTX phase totals in the report to focus prefill vs decode vs sampling.

## Global: top CUDA kernels (by total time)

- **Derived from**: `CUPTI_ACTIVITY_KIND_KERNEL`; duration = `end-start`.
- **Limitations**: totals are summed over launches (no overlap correction); names may be numeric IDs if string resolution is unavailable.

| kernel_name | device_id | total_ms | calls | avg_us | p50_us | p90_us | pct_kernel_time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn | 0 | 47.015 | 3612 | 13.02 | 12.29 | 16.86 | 9.9 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<signed char>, std::array<char *, (unsigned long)1>>(int, T2, T3) | 0 | 40.421 | 56 | 721.81 | 724.13 | 731.86 | 8.5 |
| ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 0 | 32.167 | 2940 | 10.94 | 10.11 | 12.35 | 6.8 |
| reshape_and_cache_kernel_flash | 0 | 28.818 | 5656 | 5.10 | 3.68 | 10.59 | 6.0 |
| ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn | 0 | 27.022 | 252 | 107.23 | 33.18 | 404.09 | 5.7 |
| ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_tn | 0 | 23.210 | 1232 | 18.84 | 18.94 | 19.20 | 4.9 |
| ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_32x6_tn | 0 | 22.392 | 1456 | 15.38 | 14.56 | 22.34 | 4.7 |
| ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_tn | 0 | 21.129 | 1512 | 13.97 | 12.16 | 24.35 | 4.4 |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_f16_32x32_128x2_tn_align8>(T1::Params) | 0 | 19.598 | 1904 | 10.29 | 10.27 | 10.46 | 4.1 |
| sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_cublas | 0 | 19.124 | 2128 | 8.99 | 8.74 | 10.08 | 4.0 |
| kernel_unified_attention_2d | 0 | 17.051 | 896 | 19.03 | 19.20 | 30.32 | 3.6 |
| triton_poi_fused_mul_silu_slice_1 | 0 | 14.142 | 4256 | 3.32 | 2.40 | 4.32 | 3.0 |
| ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x3_tn | 0 | 13.585 | 120 | 113.21 | 182.69 | 184.11 | 2.9 |
| triton_poi_fused_4 | 0 | 13.233 | 4104 | 3.22 | 2.69 | 3.71 | 2.8 |
| triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2 | 0 | 13.032 | 4256 | 3.06 | 2.62 | 3.68 | 2.7 |
| triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0 | 0 | 11.613 | 4256 | 2.73 | 2.43 | 3.20 | 2.4 |
| kernel_unified_attention_3d | 0 | 11.298 | 1876 | 6.02 | 6.01 | 6.24 | 2.4 |
| triton_red_fused_3 | 0 | 10.811 | 4104 | 2.63 | 2.53 | 3.07 | 2.3 |
| void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x128_32x5_tn_align8>(T1::Params) | 0 | 7.788 | 28 | 278.14 | 278.21 | 279.00 | 1.6 |
| ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_32x3_tn | 0 | 7.006 | 224 | 31.27 | 31.26 | 31.33 | 1.5 |
| reduce_segments | 0 | 6.810 | 1876 | 3.63 | 3.58 | 3.81 | 1.4 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<c10::Half>, std::array<char *, (unsigned long)1>>(int, T2, T3) | 0 | 6.794 | 3026 | 2.25 | 2.08 | 2.53 | 1.4 |
| void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __half, float, __half, (bool)0, __half, __half, __half, (bool)1, (bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 *, const T6 *, const T6 *, const T6 *, const T6 *) | 0 | 6.733 | 2408 | 2.80 | 2.69 | 3.13 | 1.4 |
| void at_cuda_detail::cub::DeviceSegmentedRadixSortKernel<at_cuda_detail::cub::DeviceRadixSortPolicy<float, long, int>::Policy900, (bool)1, (bool)0, float, long, at::native::<unnamed>::offset_t, at::native::<unnamed>::offset_t, int, at_cuda_detail::cub::detail::identity_decomposer_t>(const T4 *, T4 *, const T5 *, T5 *, T6, T7, int, int, int, T9) | 0 | 6.574 | 8 | 821.80 | 820.22 | 895.86 | 1.4 |
| sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize96x128x32_stage4_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_cublas | 0 | 5.818 | 224 | 25.97 | 25.98 | 26.14 | 1.2 |
| ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5_tn | 0 | 5.463 | 420 | 13.01 | 12.26 | 18.46 | 1.1 |
| sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize96x64x64_stage4_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_cublas | 0 | 5.267 | 308 | 17.10 | 17.47 | 17.89 | 1.1 |
| void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<char *, (unsigned long)1>>(int, T2, T3) | 0 | 5.202 | 3036 | 1.71 | 1.70 | 1.73 | 1.1 |
| void at_cuda_detail::cub::DeviceSegmentedRadixSortKernel<at_cuda_detail::cub::DeviceRadixSortPolicy<float, long, int>::Policy900, (bool)0, (bool)0, float, long, at::native::<unnamed>::offset_t, at::native::<unnamed>::offset_t, int, at_cuda_detail::cub::detail::identity_decomposer_t>(const T4 *, T4 *, const T5 *, T5 *, T6, T7, int, int, int, T9) | 0 | 4.126 | 4 | 1031.51 | 1029.95 | 1157.03 | 0.9 |
| ampere_s16816gemm_fp16_128x64_ldg8_stages_32x6_tn | 0 | 3.644 | 448 | 8.13 | 8.03 | 9.09 | 0.8 |

## Top PIDs by GPU kernel time

- **Derived from**: `CUPTI_ACTIVITY_KIND_KERNEL` grouped by PID (requires kernel PID column such as `globalPid`).
- **Limitations**: PID attribution is best-effort and depends on exported columns; missing PID columns → section unavailable.
- **PID source**: `globalPid`

| pid | total_kernel_time_ms | kernel_count | pct_of_total_kernel_time |
| --- | --- | --- | --- |
| 495200 | 476.536 | 59103 | 100.0 |

## Top kernels per PID

### PID `495200`

- PID kernel time: `476.536 ms`

| kernel_name | device_id | total_time_ms | call_count | avg_duration_us | pct_of_pid_kernel_time |
| --- | --- | --- | --- | --- | --- |
| ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn | 0 | 47.015 | 3612 | 13.02 | 9.9 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<signed char>, std::array<char *, (unsigned long)1>>(int, T2, T3) | 0 | 40.421 | 56 | 721.81 | 8.5 |
| ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn | 0 | 32.167 | 2940 | 10.94 | 6.8 |
| reshape_and_cache_kernel_flash | 0 | 28.818 | 5656 | 5.10 | 6.0 |
| ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn | 0 | 27.022 | 252 | 107.23 | 5.7 |
| ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_tn | 0 | 23.210 | 1232 | 18.84 | 4.9 |
| ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_32x6_tn | 0 | 22.392 | 1456 | 15.38 | 4.7 |
| ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_64x4_tn | 0 | 21.129 | 1512 | 13.97 | 4.4 |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_s161616gemm_f16_32x32_128x2_tn_align8>(T1::Params) | 0 | 19.598 | 1904 | 10.29 | 4.1 |
| sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize32x32x64_stage6_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_cublas | 0 | 19.124 | 2128 | 8.99 | 4.0 |

## Launch storm per PID (best-effort)

- **Derived from**: `CUPTI_ACTIVITY_KIND_KERNEL` kernel timestamps filtered by PID.
- **Limitations**: per-PID launch storm depends on PID decoding; overlap across streams does not invalidate launch rate but complicates interpretation.

| pid | total_launches | window_s | launches_per_s | p50_kernel_us | p90_kernel_us | p99_kernel_us | pct_under_5us | pct_under_10us | pct_under_20us | launch_storm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 495200 | 59103 | 83.204 | 710.340 | 3.36 | 13.31 | 32.83 | 61.3 | 74.9 | 96.3 | false |

## Sync indicators per PID

- **Derived from**: `CUPTI_ACTIVITY_KIND_RUNTIME` runtime API intervals grouped by PID (requires runtime `globalTid`/pid).
- **Limitations**: only reports what was traced/exported; some waits may not appear as explicit sync calls.
- **PID source**: `globalTid`

| pid | api_name | total_time_ms | call_count | avg_duration_us |
| --- | --- | --- | --- | --- |
| 495200 | cudaEventSynchronize_v3020 | 129.973 | 129 | 1007.54 |
| 495200 | cudaStreamSynchronize_v3020 | 72.419 | 507 | 142.84 |
| 495200 | cudaDeviceSynchronize_v3020 | 30.816 | 1552 | 19.86 |
| 495200 | cudaEventQuery_v3020 | 0.296 | 119 | 2.49 |
| 495200 | cudaStreamWaitEvent_v3020 | 0.131 | 65 | 2.01 |

## Global: launch storm

- **Derived from**: `CUPTI_ACTIVITY_KIND_KERNEL` kernel timestamps (`start/end`).
- **Limitations**: uses kernel-table window; overlap across streams doesn’t invalidate launch rate but complicates “GPU saturated” interpretation.

- launches: `59103` over `83.204s` = `710.3/s`
- duration p50/p90/p99 (us): `3.36` / `13.31` / `32.83`
- % kernels under 5/10/20 us: `61.3%` / `74.9%` / `96.3%`
- launch_storm = `False`

Top tiny kernels by call count:

| kernel_name | call_count | avg_duration_us |
| --- | --- | --- |
| triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2 | 4228 | 2.78 |
| triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0 | 4228 | 2.54 |
| triton_red_fused_3 | 4077 | 2.43 |
| triton_poi_fused_4 | 4077 | 2.88 |
| triton_poi_fused_mul_silu_slice_1 | 3947 | 2.58 |
| reshape_and_cache_kernel_flash | 3362 | 2.65 |
| void at::native::vectorized_elementwise_kernel<(int)2, at::native::FillFunctor<long>, std::array<char *, (unsigned long)1>>(int, T2, T3) | 3036 | 1.71 |
| void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<c10::Half>, std::array<char *, (unsigned long)1>>(int, T2, T3) | 2997 | 2.10 |
| void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __half, float, __half, (bool)0, __half, __half, __half, (bool)1, (bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 *, const T6 *, const T6 *, const T6 *, const T6 *) | 2408 | 2.80 |
| reduce_segments | 1854 | 3.61 |

## Global: CPU↔GPU synchronization (CUDA runtime/driver)

- **Derived from**: `CUPTI_ACTIVITY_KIND_RUNTIME` API intervals filtered by sync-like names.
- **Limitations**: only reports what was traced/exported; some waits may not appear as explicit sync calls.

| api_name | call_count | total_time_ms | avg_duration_us |
| --- | --- | --- | --- |
| cudaEventSynchronize_v3020 | 129 | 129.973 | 1007.54 |
| cudaStreamSynchronize_v3020 | 507 | 72.419 | 142.84 |
| cudaDeviceSynchronize_v3020 | 1552 | 30.816 | 19.86 |
| cudaEventQuery_v3020 | 119 | 0.296 | 2.49 |
| cudaStreamWaitEvent_v3020 | 65 | 0.131 | 2.01 |

## GPU idle estimate (from kernel timeline)

- **Derived from**: union of kernel intervals from `CUPTI_ACTIVITY_KIND_KERNEL` (per device if `deviceId` exists).
- **Limitations**: approximate/conservative; excludes memcpy/memset/other GPU activities; overlap across streams is merged (union).

| device_id | window_ms | busy_ms | idle_ms | idle_pct_of_window |
| --- | --- | --- | --- | --- |
| 0 | 83203.799 | 476.536 | 82727.263 | 99.4 |

Largest gaps:

| device_id | gap_start_ns | gap_end_ns | gap_ms |
| --- | --- | --- | --- |
| 0 | 256931692114 | 311626628394 | 54694.936 |
| 0 | 248094418808 | 251259788251 | 3165.369 |
| 0 | 253991797019 | 256928808146 | 2937.011 |
| 0 | 311626649738 | 312831647443 | 1204.998 |
| 0 | 330019055258 | 331135194254 | 1116.139 |
| 0 | 318961551192 | 320058317700 | 1096.767 |
| 0 | 325823773472 | 326647738957 | 823.965 |
| 0 | 313100094781 | 313798446454 | 698.352 |
| 0 | 320856592232 | 321517803557 | 661.211 |
| 0 | 316245296953 | 316813882923 | 568.586 |
| 0 | 315715725382 | 316245213657 | 529.488 |
| 0 | 325317030822 | 325823340832 | 506.310 |
| 0 | 320098801510 | 320599109180 | 500.308 |
| 0 | 314736159964 | 315234765137 | 498.605 |
| 0 | 316814249355 | 317292691034 | 478.442 |
| 0 | 315237173521 | 315715597542 | 478.424 |
| 0 | 323350960716 | 323825255005 | 474.294 |
| 0 | 322385257993 | 322822432921 | 437.175 |
| 0 | 329294153703 | 329729244223 | 435.091 |
| 0 | 317292916730 | 317706732551 | 413.816 |

## NVTX ranges

- **Derived from**: `NVTX_EVENTS` rows with non-null `end`, aggregated by range name.
- **Limitations**: NVTX is host-side timing; it does not directly measure GPU time without additional correlation.

| range_name | count | total_time_ms | avg_duration_us |
| --- | --- | --- | --- |
| ncclGroupStart | 14 | 19.844 | 1417.46 |
| cub::DeviceSegmentedRadixSort | 2 | 0.147 | 73.44 |
| ncclGroupEnd | 14 | 0.006 | 0.42 |


### NVTX-attributed GPU kernel time (best-effort correlationId mapping)

Coverage: `2.2%` (attributed kernel time / total kernel time) = `10.700 ms / 476.536 ms` across `12` kernels.

| range_name | kernel_count | total_kernel_time_ms | avg_kernel_duration_us | pct_of_total_kernel_time |
| --- | --- | --- | --- | --- |
| cub::DeviceSegmentedRadixSort | 12 | 10.700 | 891.70 | 2.2 |

Assumptions/limitations:
- Requires kernel `correlationId`, runtime `correlationId` + `globalTid`, and an enclosing NVTX range on the same `globalTid`.
- Attributed 2.2% of total kernel time via NVTX→runtime→kernel correlation.

## NVTX per PID (best-effort)

- **Derived from**: `NVTX_EVENTS` grouped by PID (requires NVTX `globalTid`/pid).
- **Limitations**: depends on exported NVTX columns; host-side only; GPU attribution is best-effort if present elsewhere in report.
- **PID source**: `globalTid`

| pid | range_name | nvtx_total_time_ms | nvtx_count | attributed_kernel_time_ms | attributed_kernel_count | pid_attribution_coverage_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 495200 | cub::DeviceSegmentedRadixSort | 0.147 | 2 | 10.700 | 12 | 2.2 |
| 495200 | ncclGroupStart | 19.844 | 14 | 0.000 | 0 |  |
| 495200 | ncclGroupEnd | 0.006 | 14 | 0.000 | 0 |  |

### NVTX→kernel attribution coverage by PID

| pid | pid_total_kernel_time_ms | pid_attributed_kernel_time_ms | pid_attribution_coverage_pct |
| --- | --- | --- | --- |
| 495200 | 476.536 | 10.700 | 2.2 |

## Derivation & assumptions

- **Timestamp units**: report interprets `start/end` as **nanoseconds** and converts to ms/us via `/1e6` and `/1e3`.
- **Timestamp sanity check**: `timestamp_unit_guess=ns` (basis `kernel_window_ns_ge_1s`). If `unknown`, treat time-derived numbers as suspect.
- **Kernel durations**: `end-start` from `CUPTI_ACTIVITY_KIND_KERNEL` summed over launches (no overlap correction).
- **GPU idle estimate**: per-device union of kernel intervals within the kernel time window; excludes memcpy/memset unless you extend the tool.
- **NVTX→kernel attribution**: best-effort correlation (`kernel.correlationId` → runtime launch site → `globalTid` → enclosing NVTX range). Coverage is reported; low coverage means per-phase attribution may not reflect total GPU time.
- **Per-PID attribution**: best-effort decoding from available PID-bearing columns (`pid`, `processId`, `globalPid`, `globalTid`).

