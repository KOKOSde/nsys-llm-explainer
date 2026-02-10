## Changelog

### v0.1.0

- **Offline Nsight Systems SQLite explainer**: `trace.sqlite` in → `report.md`, `report.json`, and CSV tables out.
- **Top CUDA kernels** by total time with call counts and duration stats.
- **Launch storm detection** (many tiny kernels) with percentiles and “% under X µs”.
- **CPU↔GPU sync indicators** from CUDA runtime/driver API intervals (sync-like calls and waits).
- **GPU idle gaps (estimate)** based on union of kernel intervals per device.
- **NVTX ranges** breakdown (wall time) and optional NVTX→phase mapping.
- **Best-effort NVTX-attributed GPU kernel time** via correlationId/globalTid mapping with explicit coverage reporting.
- **Multi-process (PID) breakdowns** (best-effort, when PID-bearing columns exist) for kernels, sync indicators, and NVTX.
