import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import __version__
from .heuristics import (
    findings_to_dict,
    generate_findings,
    load_phase_map,
    nvtx_kernel_phase_breakdown,
    nvtx_phase_breakdown,
)
from .queries import (
    TraceDB,
    detect_launch_storm,
    estimate_gpu_idle_gaps,
    find_sync_events,
    get_top_kernels,
    kernels_by_pid,
    nvtx_breakdown,
    nvtx_by_pid,
    nvtx_kernel_time_by_range,
    per_pid_breakdown,
    schema_discovery,
    sync_by_pid,
    write_csv,
    write_json,
)


def _fmt_ms(x: Optional[float]) -> str:
    return "-" if x is None else "{:.3f}".format(float(x))


def _fmt_us(x: Optional[float]) -> str:
    return "-" if x is None else "{:.2f}".format(float(x))


def _md_table(rows: Sequence[Mapping[str, Any]], cols: Sequence[str]) -> str:
    if not rows:
        return "_(none)_"

    def fmt_cell(col: str, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            c = col.lower()
            if "pct" in c:
                return "{:.1f}".format(v)
            if c.endswith("_ms"):
                return "{:.3f}".format(v)
            if c.endswith("_us"):
                return "{:.2f}".format(v)
            if c.endswith("_s"):
                return "{:.6f}".format(v) if abs(v) < 0.01 else "{:.3f}".format(v)
            return "{:.3f}".format(v)
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body: List[str] = []
    for r in rows:
        body.append("| " + " | ".join(fmt_cell(c, r.get(c)) for c in cols) + " |")
    return "\n".join([header, sep] + body)


@dataclass(frozen=True)
class AnalysisOutputs:
    report: Mapping[str, Any]
    markdown: str


def analyze(
    trace_db: TraceDB,
    *,
    phase_map_path: Optional[str],
    kernel_limit: int = 30,
    compute_kernel_percentiles: bool = True,
    compute_nvtx_kernel_map: bool = True,
    nvtx_coverage_warn_threshold: float = 0.70,
) -> AnalysisOutputs:
    """Run the explainer pipeline and return JSON + Markdown."""

    schema = schema_discovery(trace_db)

    top_kernels = get_top_kernels(trace_db, limit=int(kernel_limit), compute_percentiles=bool(compute_kernel_percentiles))
    launch_storm = detect_launch_storm(top_kernels, trace_db=trace_db)
    sync = find_sync_events(trace_db)
    gpu_idle = estimate_gpu_idle_gaps(trace_db, top_n_gaps=50)
    nvtx = nvtx_breakdown(trace_db)

    if compute_nvtx_kernel_map:
        nvtx_kernel = nvtx_kernel_time_by_range(trace_db, limit=50)
    else:
        nvtx_kernel = {
            "present": False,
            "ranges": [],
            "notes": ["NVTX→kernel attribution skipped (--no-nvtx-kernel-map)."],
            "sql": {},
        }

    by_pid_kernels = kernels_by_pid(trace_db, top_pids=10, top_kernels_per_pid=10, limit_pids_for_kernel_rows=10)
    by_pid_sync = sync_by_pid(trace_db, top_pids=10, limit=200)
    by_pid_nvtx = nvtx_by_pid(trace_db, limit=500) if (nvtx.get("ranges") or []) else {"present": False, "notes": ["No NVTX ranges found."], "sql": {}}
    per_pid = per_pid_breakdown(trace_db, top_pids=10, kernel_limit=10)

    phase_map = load_phase_map(phase_map_path)
    nvtx_phases: Optional[Mapping[str, Any]] = None
    if phase_map and (nvtx.get("ranges") or []):
        nvtx_phases = nvtx_phase_breakdown(nvtx, phase_map=phase_map)

    nvtx_kernel_phases: Optional[Mapping[str, Any]] = None
    if phase_map and nvtx_kernel.get("present") and (nvtx_kernel.get("ranges") or []):
        nvtx_kernel_phases = nvtx_kernel_phase_breakdown(nvtx_kernel, phase_map=phase_map)

    nvtx_kernel_phases_by_pid: Optional[List[Dict[str, Any]]] = None
    if phase_map and by_pid_nvtx.get("present"):
        kt = by_pid_nvtx.get("kernel_time_by_pid") or {}
        if kt.get("present") and (kt.get("ranges") or []):
            byp: Dict[int, List[Dict[str, Any]]] = {}
            for r in kt.get("ranges") or []:
                pid = int(r.get("pid") or -1)
                byp.setdefault(pid, []).append(r)
            nvtx_kernel_phases_by_pid = []
            for pid, ranges in sorted(byp.items(), key=lambda kv: kv[0]):
                phases = nvtx_kernel_phase_breakdown({"present": True, "ranges": ranges}, phase_map=phase_map)
                nvtx_kernel_phases_by_pid.append({"pid": pid, "phases": phases.get("phases") or []})

    warnings: List[str] = []
    if nvtx_kernel.get("present") and compute_nvtx_kernel_map:
        cov = float(nvtx_kernel.get("coverage_fraction") or 0.0)
        if cov < float(nvtx_coverage_warn_threshold):
            warnings.append(
                "NVTX-attributed GPU time is best-effort (NVTX→runtime→kernel correlation). "
                "Coverage is {:.1f}% (< {:.1f}%). Low coverage → interpret cautiously.".format(
                    float(nvtx_kernel.get("coverage_pct") or 0.0), float(nvtx_coverage_warn_threshold) * 100.0
                )
            )

    # Per-PID NVTX→kernel coverage warnings (if present).
    try:
        kt = (by_pid_nvtx.get("kernel_time_by_pid") or {}) if isinstance(by_pid_nvtx, dict) else {}
        pid_cov_rows = kt.get("pids") or []
        low_cov = []
        for r in pid_cov_rows:
            cov = float(r.get("pid_attribution_coverage_fraction") or 0.0)
            if cov < float(nvtx_coverage_warn_threshold):
                low_cov.append((int(r.get("pid") or -1), float(r.get("pid_attribution_coverage_pct") or 0.0)))
        if low_cov:
            worst = sorted(low_cov, key=lambda t: t[1])[0]
            warnings.append(
                "Per-PID NVTX-attributed GPU time has low coverage for at least one PID (worst PID {}: {:.1f}%). "
                "Interpret per-phase/per-PID attribution cautiously.".format(int(worst[0]), float(worst[1]))
            )
    except Exception:
        pass

    # PID plausibility warnings (best-effort).
    pid_attr: Dict[str, Any] = {}
    try:
        kp = by_pid_kernels.get("pids") or []
        rp = by_pid_sync.get("pids") or []
        np = []
        try:
            # Prefer per-PID attribution coverage list if present; otherwise infer from nvtx rows.
            kt = (by_pid_nvtx.get("kernel_time_by_pid") or {})
            np = kt.get("pids") or []
        except Exception:
            np = []

        def uniq_pids(rows: Sequence[Mapping[str, Any]]) -> List[int]:
            out = sorted({int(r.get("pid")) for r in rows if r.get("pid") is not None and int(r.get("pid")) >= 0})
            return out

        k_pids = uniq_pids(kp)
        r_pids = uniq_pids(rp)
        n_pids = uniq_pids(np)
        pid_attr = {
            "kernel_pid_source": by_pid_kernels.get("pid_source"),
            "runtime_pid_source": by_pid_sync.get("pid_source"),
            "nvtx_pid_source": (by_pid_nvtx.get("pid_source") if isinstance(by_pid_nvtx, dict) else None),
            "kernel_pid_count": len(k_pids),
            "runtime_pid_count": len(r_pids),
            "nvtx_pid_count": len(n_pids),
            "kernel_pids_sample": k_pids[:10],
            "runtime_pids_sample": r_pids[:10],
            "nvtx_pids_sample": n_pids[:10],
        }

        suspicious = False
        if k_pids and all(pid == 0 for pid in k_pids):
            suspicious = True
        if k_pids and max(k_pids) > 10_000_000:
            suspicious = True
        # If runtime or NVTX shows multiple PIDs but kernels only show one, warn.
        if len(k_pids) == 1 and (len(r_pids) > 1 or len(n_pids) > 1):
            suspicious = True
        if suspicious:
            warnings.append(
                "PID attribution may be unavailable/ambiguous for this trace export. "
                "Kernel PID source=`{}`; distinct PIDs observed: kernels={}, runtime={}, nvtx={}. "
                "Interpret per-PID sections cautiously.".format(
                    str(by_pid_kernels.get("pid_source")),
                    len(k_pids),
                    len(r_pids),
                    len(n_pids),
                )
            )
    except Exception:
        pid_attr = {"present": False}

    findings = generate_findings(
        top_kernels=top_kernels,
        launch_storm=launch_storm,
        sync=sync,
        gpu_idle=gpu_idle,
        nvtx=nvtx,
        nvtx_phases=nvtx_phases,
        nvtx_kernel_phases=nvtx_kernel_phases,
    )

    report: Dict[str, Any] = {
        "tool": {"name": "nsys-llm-explain", "version": __version__},
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "trace": {"path": str(trace_db.path)},
        "warnings": warnings,
        "schema": schema,
        "metrics": {
            "top_kernels": top_kernels,
            "launch_storm": launch_storm,
            "sync": sync,
            "gpu_idle": gpu_idle,
            "nvtx": nvtx,
            "nvtx_phases": nvtx_phases,
            "nvtx_kernel_time": nvtx_kernel,
            "nvtx_kernel_phases": nvtx_kernel_phases,
            "nvtx_coverage_warn_threshold": float(nvtx_coverage_warn_threshold),
            "per_pid": per_pid,
            "pid_attribution": pid_attr,
            "by_pid": {
                "kernels": by_pid_kernels,
                "sync": by_pid_sync,
                "nvtx": by_pid_nvtx,
                "nvtx_kernel_phases": nvtx_kernel_phases_by_pid,
            },
        },
        "findings": findings_to_dict(findings),
    }

    md = render_markdown(report)
    return AnalysisOutputs(report=report, markdown=md)


def write_artifacts(outputs: AnalysisOutputs, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    write_json(outputs.report, out_dir / "report.json")
    (out_dir / "report.md").write_text(outputs.markdown, encoding="utf-8")

    m = outputs.report["metrics"]
    write_csv(m["top_kernels"].get("kernels") or [], tables_dir / "kernels.csv")
    write_csv(m["gpu_idle"].get("gaps") or [], tables_dir / "gpu_idle_gaps.csv")

    if m.get("nvtx") and (m["nvtx"].get("ranges") or []):
        write_csv(m["nvtx"].get("ranges") or [], tables_dir / "nvtx_ranges.csv")
    else:
        write_csv([], tables_dir / "nvtx_ranges.csv")

    by_pid = m.get("by_pid") or {}
    write_csv(((by_pid.get("kernels") or {}).get("kernels") or []), tables_dir / "kernels_by_pid.csv")
    write_csv(((by_pid.get("sync") or {}).get("sync_calls") or []), tables_dir / "sync_by_pid.csv")

    nb = by_pid.get("nvtx") or {}
    if nb.get("present") and (nb.get("ranges") or []):
        write_csv(nb.get("ranges") or [], tables_dir / "nvtx_by_pid.csv")


def render_markdown(report: Mapping[str, Any]) -> str:
    m = report["metrics"]
    schema = report.get("schema") or {}

    lines: List[str] = []
    lines.append("# Nsight Systems LLM Hotspot Report")
    lines.append("")
    lines.append("- Generated at (UTC): `{}`".format(report.get("generated_at")))
    lines.append("- Trace: `{}`".format(report.get("trace", {}).get("path")))
    lines.append("- Tool: `nsys-llm-explain {}`".format(report.get("tool", {}).get("version")))
    lines.append("")

    warnings = report.get("warnings") or []
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append("- {}".format(w))
        lines.append("")

    lines.append("## What to do next")
    lines.append("")
    findings = report.get("findings") or []
    if not findings:
        lines.append("_No findings generated._")
    else:
        for f in findings:
            lines.append("- **[{sev}] {title}**".format(sev=f.get("severity", "low"), title=f.get("title", "")))
            ev = f.get("evidence") or []
            rec = f.get("recommendation") or []
            if ev:
                lines.append("  - **Evidence**:")
                for e in ev:
                    lines.append("    - {}".format(e))
            if rec:
                lines.append("  - **Recommendation**:")
                for r in rec:
                    lines.append("    - {}".format(r))
    lines.append("")

    lines.append("## Global: top CUDA kernels (by total time)")
    lines.append("")
    lines.append("- **Derived from**: `{}`; duration = `end-start`.".format(m["top_kernels"].get("table")))
    lines.append("- **Limitations**: totals are summed over launches (no overlap correction); names may be numeric IDs if string resolution is unavailable.")
    lines.append("")
    kernels = m["top_kernels"].get("kernels") or []
    krows: List[Dict[str, Any]] = []
    for k in kernels[:30]:
        krows.append(
            {
                "kernel_name": k.get("kernel_name"),
                "device_id": k.get("device_id"),
                "total_ms": _fmt_ms(k.get("total_time_ms")),
                "calls": k.get("call_count"),
                "avg_us": _fmt_us(k.get("avg_duration_us")),
                "p50_us": _fmt_us(k.get("p50_duration_us")),
                "p90_us": _fmt_us(k.get("p90_duration_us")),
                "pct_kernel_time": "{:.1f}".format(float(k.get("pct_total_kernel_time") or 0.0)),
            }
        )
    lines.append(_md_table(krows, cols=["kernel_name", "device_id", "total_ms", "calls", "avg_us", "p50_us", "p90_us", "pct_kernel_time"]))
    lines.append("")

    by_pid = m.get("by_pid") or {}
    kb = by_pid.get("kernels") or {}

    lines.append("## Top PIDs by GPU kernel time")
    lines.append("")
    lines.append("- **Derived from**: `{}` grouped by PID (requires kernel PID column such as `globalPid`).".format(schema.get("kernel_table")))
    lines.append("- **Limitations**: PID attribution is best-effort and depends on exported columns; missing PID columns → section unavailable.")
    lines.append("- **PID source**: `{}`".format(kb.get("pid_source")))
    lines.append("")
    if kb.get("present") and (kb.get("pids") or []):
        lines.append(_md_table(kb.get("pids")[:20], cols=["pid", "total_kernel_time_ms", "kernel_count", "pct_of_total_kernel_time"]))
    else:
        lines.append("_(PID breakdown unavailable for kernels on this export.)_")
    lines.append("")

    lines.append("## Top kernels per PID")
    lines.append("")
    if kb.get("present") and (kb.get("kernels") or []):
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for r in kb.get("kernels") or []:
            grouped.setdefault(int(r.get("pid") or -1), []).append(r)
        pid_totals: Dict[int, float] = {}
        for r in (kb.get("pids") or []):
            try:
                pid_totals[int(r.get("pid") or -1)] = float(r.get("total_kernel_time_ms") or 0.0)
            except Exception:
                pass
        for pid, rows in sorted(grouped.items(), key=lambda kv: kv[0]):
            lines.append("### PID `{}`".format(pid))
            lines.append("")
            if pid in pid_totals:
                lines.append("- PID kernel time: `{:.3f} ms`".format(pid_totals[pid]))
                lines.append("")
            lines.append(
                _md_table(
                    rows[:20],
                    cols=["kernel_name", "device_id", "total_time_ms", "call_count", "avg_duration_us", "pct_of_pid_kernel_time"],
                )
            )
            lines.append("")
    else:
        lines.append("_(no per-PID kernel rows)_")
        lines.append("")

    lines.append("## Launch storm per PID (best-effort)")
    lines.append("")
    lines.append("- **Derived from**: `{}` kernel timestamps filtered by PID.".format(schema.get("kernel_table")))
    lines.append("- **Limitations**: per-PID launch storm depends on PID decoding; overlap across streams does not invalidate launch rate but complicates interpretation.")
    lines.append("")
    pp = m.get("per_pid") or {}
    if pp.get("present") and (pp.get("pids") or []):
        rows: List[Dict[str, Any]] = []
        for p in pp.get("pids") or []:
            pid = p.get("pid")
            ls = p.get("launch_storm") or {}
            rows.append(
                {
                    "pid": pid,
                    "total_launches": int(ls.get("total_launches") or 0),
                    "window_s": float(ls.get("window_s") or 0.0),
                    "launches_per_s": float(ls.get("launches_per_s") or 0.0),
                    "p50_kernel_us": ls.get("p50_kernel_us") if ls.get("p50_kernel_us") is not None else ls.get("median_kernel_us"),
                    "p90_kernel_us": ls.get("p90_kernel_us"),
                    "p99_kernel_us": ls.get("p99_kernel_us"),
                    "pct_under_5us": ls.get("pct_under_5us"),
                    "pct_under_10us": ls.get("pct_under_10us"),
                    "pct_under_20us": ls.get("pct_under_20us"),
                    "launch_storm": ls.get("is_launch_storm"),
                }
            )
        rows.sort(key=lambda r: float(r.get("launches_per_s") or 0.0), reverse=True)
        lines.append(
            _md_table(
                rows[:50],
                cols=[
                    "pid",
                    "total_launches",
                    "window_s",
                    "launches_per_s",
                    "p50_kernel_us",
                    "p90_kernel_us",
                    "p99_kernel_us",
                    "pct_under_5us",
                    "pct_under_10us",
                    "pct_under_20us",
                    "launch_storm",
                ],
            )
        )
    else:
        lines.append("_(PID breakdown unavailable for launch storm on this export.)_")
    lines.append("")

    lines.append("## Sync indicators per PID")
    lines.append("")
    lines.append("- **Derived from**: `{}` runtime API intervals grouped by PID (requires runtime `globalTid`/pid).".format(schema.get("runtime_table")))
    lines.append("- **Limitations**: only reports what was traced/exported; some waits may not appear as explicit sync calls.")
    sb = by_pid.get("sync") or {}
    lines.append("- **PID source**: `{}`".format(sb.get("pid_source")))
    lines.append("")
    if sb.get("present") and (sb.get("sync_calls") or []):
        lines.append(_md_table(sb.get("sync_calls")[:50], cols=["pid", "api_name", "total_time_ms", "call_count", "avg_duration_us"]))
    else:
        lines.append("_(PID breakdown unavailable for sync calls on this export.)_")
    lines.append("")

    lines.append("## Global: launch storm")
    lines.append("")
    lines.append("- **Derived from**: `{}` kernel timestamps (`start/end`).".format(schema.get("kernel_table")))
    lines.append("- **Limitations**: uses kernel-table window; overlap across streams doesn’t invalidate launch rate but complicates “GPU saturated” interpretation.")
    lines.append("")
    ls = m["launch_storm"]
    lines.append(
        "- launches: `{}` over `{:.3f}s` = `{:.1f}/s`".format(
            int(ls.get("total_launches") or 0),
            float(ls.get("window_s") or 0.0),
            float(ls.get("launches_per_s") or 0.0),
        )
    )
    if ls.get("p50_kernel_us") is not None:
        lines.append(
            "- duration p50/p90/p99 (us): `{}` / `{}` / `{}`".format(
                _fmt_us(ls.get("p50_kernel_us")),
                _fmt_us(ls.get("p90_kernel_us")),
                _fmt_us(ls.get("p99_kernel_us")),
            )
        )
    if ls.get("pct_under_10us") is not None:
        lines.append(
            "- % kernels under 5/10/20 us: `{:.1f}%` / `{:.1f}%` / `{:.1f}%`".format(
                float(ls.get("pct_under_5us") or 0.0),
                float(ls.get("pct_under_10us") or 0.0),
                float(ls.get("pct_under_20us") or 0.0),
            )
        )
    if ls.get("is_launch_storm") is not None:
        lines.append("- launch_storm = `{}`".format(bool(ls.get("is_launch_storm"))))
    tiny = ls.get("tiny_kernels") or []
    if tiny:
        lines.append("")
        lines.append("Top tiny kernels by call count:")
        lines.append("")
        lines.append(_md_table(tiny, cols=["kernel_name", "call_count", "avg_duration_us"]))
    lines.append("")

    lines.append("## Global: CPU↔GPU synchronization (CUDA runtime/driver)")
    lines.append("")
    lines.append("- **Derived from**: `{}` API intervals filtered by sync-like names.".format(m["sync"].get("table")))
    lines.append("- **Limitations**: only reports what was traced/exported; some waits may not appear as explicit sync calls.")
    lines.append("")
    sync_calls = m["sync"].get("sync_calls") or []
    if not sync_calls:
        lines.append("_(none detected)_")
    else:
        lines.append(_md_table(sync_calls[:30], cols=["api_name", "call_count", "total_time_ms", "avg_duration_us"]))
    lines.append("")

    lines.append("## GPU idle estimate (from kernel timeline)")
    lines.append("")
    lines.append("- **Derived from**: union of kernel intervals from `{}` (per device if `deviceId` exists).".format(m["gpu_idle"].get("table")))
    lines.append("- **Limitations**: approximate/conservative; excludes memcpy/memset/other GPU activities; overlap across streams is merged (union).")
    lines.append("")
    devices = m["gpu_idle"].get("devices") or []
    if devices:
        lines.append(_md_table(devices, cols=["device_id", "window_ms", "busy_ms", "idle_ms", "idle_pct_of_window"]))
    else:
        lines.append("_(no kernel activity to estimate idle)_")
    lines.append("")

    gaps = m["gpu_idle"].get("gaps") or []
    if gaps:
        lines.append("Largest gaps:")
        lines.append("")
        lines.append(_md_table(gaps[:20], cols=["device_id", "gap_start_ns", "gap_end_ns", "gap_ms"]))
        lines.append("")

    lines.append("## NVTX ranges")
    lines.append("")
    lines.append("- **Derived from**: `{}` rows with non-null `end`, aggregated by range name.".format(m["nvtx"].get("table")))
    lines.append("- **Limitations**: NVTX is host-side timing; it does not directly measure GPU time without additional correlation.")
    lines.append("")
    nv = m["nvtx"]
    if not nv.get("ranges"):
        lines.append("_(no NVTX ranges found)_")
    else:
        lines.append(_md_table(nv.get("ranges")[:30], cols=["range_name", "count", "total_time_ms", "avg_duration_us"]))
        lines.append("")

        phases = m.get("nvtx_phases") or {}
        if phases and phases.get("phases"):
            lines.append("### NVTX phases (mapped, wall-time)")
            lines.append("")
            lines.append(_md_table(phases.get("phases")[:20], cols=["phase", "total_time_ms", "pct_of_nvtx_total"]))

        nvk = m.get("nvtx_kernel_time") or {}
        if nvk.get("present") and (nvk.get("ranges") or []):
            lines.append("")
            lines.append("### NVTX-attributed GPU kernel time (best-effort correlationId mapping)")
            lines.append("")
            tot_ms = float(nvk.get("total_kernel_time_ns") or 0) / 1_000_000.0
            mapped_ms = float(nvk.get("mapped_kernel_time_ms") or 0.0)
            mapped_k = int(nvk.get("mapped_kernel_count") or 0)
            lines.append(
                "Coverage: `{:.1f}%` (attributed kernel time / total kernel time) = `{:.3f} ms / {:.3f} ms` across `{}` kernels.".format(
                    float(nvk.get("coverage_pct") or 0.0), mapped_ms, tot_ms, mapped_k
                )
            )
            lines.append("")
            lines.append(
                _md_table(
                    nvk.get("ranges")[:30],
                    cols=["range_name", "kernel_count", "total_kernel_time_ms", "avg_kernel_duration_us", "pct_of_total_kernel_time"],
                )
            )
            notes = nvk.get("notes") or []
            if notes:
                lines.append("")
                lines.append("Assumptions/limitations:")
                lines.append("- Requires kernel `correlationId`, runtime `correlationId` + `globalTid`, and an enclosing NVTX range on the same `globalTid`.")
                for n in notes:
                    lines.append("- {}".format(n))

        kp = m.get("nvtx_kernel_phases") or {}
        if kp and kp.get("phases"):
            lines.append("")
            lines.append("### NVTX phases (mapped, GPU kernel time)")
            lines.append("")
            lines.append(_md_table(kp.get("phases")[:20], cols=["phase", "total_kernel_time_ms", "pct_of_attributed_kernel_time"]))

    lines.append("")

    lines.append("## NVTX per PID (best-effort)")
    lines.append("")
    lines.append("- **Derived from**: `{}` grouped by PID (requires NVTX `globalTid`/pid).".format(schema.get("nvtx_table")))
    lines.append("- **Limitations**: depends on exported NVTX columns; host-side only; GPU attribution is best-effort if present elsewhere in report.")
    nb = by_pid.get("nvtx") or {}
    lines.append("- **PID source**: `{}`".format(nb.get("pid_source")))
    lines.append("")
    if nb.get("present") and (nb.get("ranges") or []):
        lines.append(
            _md_table(
                nb.get("ranges")[:60],
                cols=[
                    "pid",
                    "range_name",
                    "nvtx_total_time_ms",
                    "nvtx_count",
                    "attributed_kernel_time_ms",
                    "attributed_kernel_count",
                    "pid_attribution_coverage_pct",
                ],
            )
        )

        kt = nb.get("kernel_time_by_pid") or {}
        if kt.get("present") and (kt.get("pids") or []):
            lines.append("")
            lines.append("### NVTX→kernel attribution coverage by PID")
            lines.append("")
            lines.append(
                _md_table(
                    kt.get("pids")[:20],
                    cols=["pid", "pid_total_kernel_time_ms", "pid_attributed_kernel_time_ms", "pid_attribution_coverage_pct"],
                )
            )

        kph = by_pid.get("nvtx_kernel_phases") or []
        if kph:
            lines.append("")
            lines.append("### NVTX phases per PID (mapped, GPU kernel time)")
            lines.append("")
            for row in kph:
                pid = row.get("pid")
                phases = row.get("phases") or []
                lines.append("#### PID `{}`".format(pid))
                lines.append("")
                lines.append(_md_table(phases[:10], cols=["phase", "total_kernel_time_ms", "pct_of_attributed_kernel_time"]))
                lines.append("")
    else:
        lines.append("_(no NVTX PID breakdown available)_")
    lines.append("")

    return "\n".join(lines) + "\n"
