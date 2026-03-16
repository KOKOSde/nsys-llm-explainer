## Changelog

### v0.2.0

- **Dashboard expansion**:
  - Added side-by-side baseline comparison (upload current + baseline trace/report).
  - Added kernel waterfall, roofline scatter, top-50 timeline, NCCL summary, NVLink timeline, stream overlap, launch latency, phase split, and per-rank NCCL skew panels in one dashboard flow.
  - Added one-click static dashboard export to HTML.
- **New analytics in `report.json` + CSVs**:
  - Timeline event extraction (`timeline_events.csv`).
  - Copy-engine extraction (`copy_engine_events.csv`).
  - Launch-latency rows + histogram (`launch_latency_rows.csv`, `launch_latency_histogram.csv`).
  - Stream-overlap summary (`stream_overlap.csv`).
  - Phase split (`phase_split.csv`) and roofline metrics (`roofline.csv`).
  - NCCL rank skew export (`nccl_rank_skew.csv`) and NVLink timeseries export (`nvlink_timeseries.csv`).
- **NVLink handling improvements**:
  - Added report-time downsampling for large NVLink timeseries to keep outputs readable and dashboard-friendly.
- **Docs refresh**:
  - Updated README with release status and dashboard screenshots.
  - Replaced placeholder dashboard section with concrete feature coverage and panel examples.

### v0.1.0

- **Offline Nsight Systems SQLite explainer**: `trace.sqlite` in → `report.md`, `report.json`, and CSV tables out.
- **Top CUDA kernels** by total time with call counts and duration stats.
- **Launch storm detection** (many tiny kernels) with percentiles and “% under X µs”.
- **CPU↔GPU sync indicators** from CUDA runtime/driver API intervals (sync-like calls and waits).
- **GPU idle gaps (estimate)** based on union of kernel intervals per device.
- **NVTX ranges** breakdown (wall time) and optional NVTX→phase mapping.
- **Best-effort NVTX-attributed GPU kernel time** via correlationId/globalTid mapping with explicit coverage reporting.
- **Multi-process (PID) breakdowns** (best-effort, when PID-bearing columns exist) for kernels, sync indicators, and NVTX.
