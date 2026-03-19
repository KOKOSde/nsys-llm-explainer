from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _bootstrap_src_path() -> None:
    here = Path(__file__).resolve()
    candidates = [parent / "src" for parent in (here.parent, *tuple(here.parents))]
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


_bootstrap_src_path()

from nsys_llm_explainer.queries import TraceDB  # type: ignore
from nsys_llm_explainer.report import AnalysisOutputs, analyze, render_markdown, write_artifacts  # type: ignore


@dataclass(frozen=True)
class SpaceBundle:
    source_path: Path
    source_kind: str
    report: Dict[str, Any]
    markdown: str
    artifacts_dir: Path
    artifact_paths: List[Path]
    summary_rows: List[Dict[str, str]]
    manifest_rows: List[Dict[str, str]]
    findings_markdown: str
    status_markdown: str


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_text(value: Any, default: str = "-") -> str:
    text = str(value).strip() if value is not None else ""
    return text if text else default


def _safe_trace_name(report: Mapping[str, Any]) -> str:
    trace_path = ((report.get("trace") or {}).get("path") or report.get("_source_name") or "")
    return Path(str(trace_path)).name if trace_path else "unknown"


def _top_kernel_row(report: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    rows = ((report.get("metrics") or {}).get("top_kernels") or {}).get("kernels") or []
    return rows[0] if rows else None


def _top_nccl_row(report: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    rows = ((report.get("metrics") or {}).get("nccl") or {}).get("ops") or []
    return rows[0] if rows else None


def _format_ms(value: Any) -> str:
    return "{:.3f} ms".format(_coerce_float(value))


def _format_us(value: Any) -> str:
    return "{:.2f} us".format(_coerce_float(value))


def _format_pct(value: Any) -> str:
    return "{:.1f}%".format(_coerce_float(value))


def _bottleneck_sentence(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics") or {}
    total_gpu_ms = _coerce_float((metrics.get("top_kernels") or {}).get("total_kernel_time_ns")) / 1_000_000.0
    top_kernel = _top_kernel_row(report)
    top_nccl = _top_nccl_row(report)
    if total_gpu_ms > 0.0 and top_nccl:
        nccl_pct = (_coerce_float(top_nccl.get("total_time_ms")) / total_gpu_ms) * 100.0
        kernel_pct = _coerce_float(top_kernel.get("pct_total_kernel_time") if top_kernel else 0.0)
        if nccl_pct >= kernel_pct:
            return "{} dominates {:.1f}% of GPU time".format(str(top_nccl.get("op_name") or "NCCL"), nccl_pct)
    if top_kernel:
        return "{} dominates {:.1f}% of GPU time".format(
            str(top_kernel.get("kernel_name") or "Top kernel"),
            _coerce_float(top_kernel.get("pct_total_kernel_time")),
        )
    return "No dominant GPU bottleneck detected from available metrics"


def _summary_rows(report: Mapping[str, Any]) -> List[Dict[str, str]]:
    metrics = report.get("metrics") or {}
    timeline = metrics.get("timeline") or {}
    gpu_total_ms = _coerce_float(timeline.get("total_gpu_time_ms"))
    if gpu_total_ms <= 0:
        gpu_total_ms = _coerce_float((metrics.get("top_kernels") or {}).get("total_kernel_time_ns")) / 1_000_000.0
    cpu_total_ms = _coerce_float(timeline.get("total_cpu_time_ms"))
    if cpu_total_ms <= 0:
        sync_rows = (metrics.get("sync") or {}).get("sync_calls") or []
        cpu_total_ms = sum(_coerce_float(row.get("total_time_ms")) for row in sync_rows)

    warnings = report.get("warnings") or []
    report_version = _safe_text((report.get("tool") or {}).get("version"), default="unknown")
    top_kernel = _top_kernel_row(report)
    top_nccl = _top_nccl_row(report)
    nvlink = (metrics.get("nvlink_during_nccl") or {}).get("rows") or []
    nvlink_row = nvlink[0] if nvlink else None
    capability_checks = {
        "Kernel table": bool((metrics.get("top_kernels") or {}).get("present")),
        "Runtime table": bool((metrics.get("sync") or {}).get("present")),
        "NVTX ranges": bool((metrics.get("nvtx") or {}).get("present")),
        "GPU metrics": bool((metrics.get("nvlink_during_nccl") or {}).get("present")),
        "Per-process breakdown": bool((metrics.get("per_pid") or {}).get("present")),
    }

    rows: List[Dict[str, str]] = [
        {"section": "Overview", "metric": "Trace", "value": _safe_trace_name(report)},
        {"section": "Overview", "metric": "Tool version", "value": report_version},
        {"section": "Overview", "metric": "Generated at (UTC)", "value": _safe_text(report.get("generated_at"))},
        {"section": "Overview", "metric": "Total GPU time", "value": _format_ms(gpu_total_ms)},
        {"section": "Overview", "metric": "Total CPU time", "value": _format_ms(cpu_total_ms)},
        {"section": "Overview", "metric": "Top bottleneck", "value": _bottleneck_sentence(report)},
        {"section": "Overview", "metric": "Warnings", "value": str(len(warnings))},
    ]
    if top_kernel:
        rows.extend(
            [
                {"section": "Evidence", "metric": "Top kernel", "value": _safe_text(top_kernel.get("kernel_name"))},
                {"section": "Evidence", "metric": "Top kernel time", "value": _format_ms(top_kernel.get("total_time_ms"))},
                {"section": "Evidence", "metric": "Top kernel share", "value": _format_pct(top_kernel.get("pct_total_kernel_time"))},
            ]
        )
    if top_nccl:
        rows.extend(
            [
                {"section": "Evidence", "metric": "Top NCCL op", "value": _safe_text(top_nccl.get("op_name"))},
                {"section": "Evidence", "metric": "Top NCCL time", "value": _format_ms(top_nccl.get("total_time_ms"))},
                {"section": "Evidence", "metric": "Top NCCL overlap", "value": _format_pct(top_nccl.get("compute_overlap_pct"))},
            ]
        )
    if nvlink_row:
        rows.extend(
            [
                {"section": "Evidence", "metric": "NVLink metric(s)", "value": _safe_text(nvlink_row.get("metric_names"))},
                {
                    "section": "Evidence",
                    "metric": "NVLink during NCCL",
                    "value": "{:.2f} export units".format(_coerce_float(nvlink_row.get("avg_metric_during_nccl"), 0.0)),
                },
                {
                    "section": "Evidence",
                    "metric": "NVLink outside NCCL",
                    "value": "{:.2f} export units".format(_coerce_float(nvlink_row.get("avg_metric_outside_nccl"), 0.0)),
                },
                {
                    "section": "Evidence",
                    "metric": "NVLink correlation",
                    "value": "{:.3f}".format(_coerce_float(nvlink_row.get("nccl_activity_correlation"), 0.0)),
                },
            ]
        )
    for label, present in capability_checks.items():
        rows.append({"section": "Capabilities", "metric": label, "value": "present" if present else "missing"})
    return rows


def _findings_markdown(report: Mapping[str, Any]) -> str:
    findings = report.get("findings") or []
    warnings = report.get("warnings") or []

    lines: List[str] = ["## What to do next", ""]
    if not findings:
        lines.append("No findings were generated for this trace.")
    else:
        for finding in findings:
            severity = _safe_text(finding.get("severity"), default="unknown").upper()
            title = _safe_text(finding.get("title"), default="Untitled finding")
            lines.append("### [{}] {}".format(severity, title))
            evidence = finding.get("evidence") or []
            recommendations = finding.get("recommendation") or finding.get("recommendations") or []
            if evidence:
                lines.append("Evidence:")
                for item in evidence:
                    lines.append("- {}".format(item))
            if recommendations:
                lines.append("Recommendation:")
                if isinstance(recommendations, (list, tuple)):
                    for item in recommendations:
                        lines.append("- {}".format(item))
                else:
                    lines.append("- {}".format(recommendations))
            lines.append("")
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append("- {}".format(warning))
    return "\n".join(lines).strip()


def _artifact_manifest(out_dir: Path) -> List[Dict[str, str]]:
    purpose_map = {
        "report.md": "Human-readable report",
        "report.json": "Machine-readable report",
        "kernels.csv": "Top kernels",
        "barriers.csv": "CPU/GPU barriers",
        "nccl_ops.csv": "Top NCCL ops",
        "nccl_rank_skew.csv": "Per-rank NCCL skew",
        "nccl_by_pid.csv": "NCCL per PID",
        "nvlink_during_nccl.csv": "NVLink correlation rows",
        "nvlink_timeseries.csv": "NVLink correlation timeseries",
        "timeline_events.csv": "Timeline events",
        "copy_engine_events.csv": "Copy engine events",
        "launch_latency_rows.csv": "Launch latency rows",
        "launch_latency_histogram.csv": "Launch latency histogram",
        "stream_overlap.csv": "Stream overlap summary",
        "phase_split.csv": "Phase split",
        "roofline.csv": "Roofline rows",
        "gpu_idle_gaps.csv": "GPU idle gaps",
        "kernels_by_pid.csv": "Per-PID kernels",
        "sync_by_pid.csv": "Per-PID sync calls",
        "nvtx_by_pid.csv": "Per-PID NVTX ranges",
        "nvtx_ranges.csv": "NVTX ranges",
        "bundle.zip": "Download all artifacts as a zip",
    }
    rows: List[Dict[str, str]] = []
    for name, purpose in purpose_map.items():
        path = out_dir / name
        if not path.exists():
            path = out_dir / "tables" / name
        if path.exists():
            rows.append({"artifact": name, "purpose": purpose, "path": str(path)})
    return rows


def _zip_artifacts(out_dir: Path) -> Path:
    zip_path = out_dir / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, arcname=path.relative_to(out_dir).as_posix())
    return zip_path


def _normalize_report_for_artifacts(report: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = dict(report)
    metrics: Dict[str, Any] = dict(normalized.get("metrics") or {})

    metrics.setdefault("top_kernels", {"present": False, "kernels": []})
    metrics.setdefault("barriers", {"present": False, "barriers": []})
    metrics.setdefault("nccl", {"present": False, "ops": [], "rank_rows": [], "pids": []})
    metrics.setdefault("nvlink_during_nccl", {"present": False, "rows": [], "timeseries": []})
    metrics.setdefault("timeline", {"present": False, "events": []})
    metrics.setdefault("copy_engine", {"present": False, "events": []})
    metrics.setdefault("launch_latency", {"present": False, "rows": [], "histogram": []})
    metrics.setdefault("stream_overlap", {"present": False, "summary": []})
    metrics.setdefault("phase_split", {"present": False, "rows": []})
    metrics.setdefault("roofline", {"present": False, "rows": []})
    metrics.setdefault("gpu_idle", {"present": False, "gaps": []})
    metrics.setdefault("nvtx", {"present": False, "ranges": []})

    by_pid = dict(metrics.get("by_pid") or {})
    by_pid.setdefault("kernels", {"kernels": []})
    by_pid.setdefault("sync", {"sync_calls": []})
    by_pid.setdefault("nvtx", {"present": False, "ranges": []})
    metrics["by_pid"] = by_pid

    normalized["metrics"] = metrics
    return normalized


def _load_report(path: Path) -> Tuple[str, Dict[str, Any], str]:
    lower = path.suffix.lower()
    if lower in (".sqlite", ".db"):
        db = TraceDB.open(path)
        try:
            outputs = analyze(
                db,
                phase_map_path=None,
                kernel_limit=50,
                compute_kernel_percentiles=True,
                compute_nvtx_kernel_map=True,
            )
            return "sqlite", dict(outputs.report), str(outputs.markdown)
        finally:
            db.close()
    if lower == ".json":
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("Input JSON root must be an object.")
        try:
            markdown = render_markdown(report)
        except Exception:
            markdown = "# Nsight Systems LLM Hotspot Report\n\nJSON loaded, but markdown rendering failed for this input."
        return "json", report, markdown
    header = path.read_bytes()[:32]
    if header.startswith(b"SQLite format 3"):
        db = TraceDB.open(path)
        try:
            outputs = analyze(
                db,
                phase_map_path=None,
                kernel_limit=50,
                compute_kernel_percentiles=True,
                compute_nvtx_kernel_map=True,
            )
            return "sqlite", dict(outputs.report), str(outputs.markdown)
        finally:
            db.close()
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Input JSON root must be an object.")
    try:
        markdown = render_markdown(report)
    except Exception:
        markdown = "# Nsight Systems LLM Hotspot Report\n\nJSON loaded, but markdown rendering failed for this input."
    return "json", report, markdown


def analyze_path(path: Path) -> SpaceBundle:
    source_kind, report, markdown = _load_report(path)
    report = _normalize_report_for_artifacts(report)
    outputs = AnalysisOutputs(report=report, markdown=markdown)
    artifacts_dir = Path(tempfile.mkdtemp(prefix="nsys-llm-explainer-space-")) / path.stem
    write_artifacts(outputs, artifacts_dir)
    _zip_artifacts(artifacts_dir)
    artifact_paths = sorted(
        [p for p in artifacts_dir.rglob("*") if p.is_file()],
        key=lambda item: item.relative_to(artifacts_dir).as_posix(),
    )
    return SpaceBundle(
        source_path=path,
        source_kind=source_kind,
        report=report,
        markdown=markdown,
        artifacts_dir=artifacts_dir,
        artifact_paths=artifact_paths,
        summary_rows=_summary_rows(report),
        manifest_rows=_artifact_manifest(artifacts_dir),
        findings_markdown=_findings_markdown(report),
        status_markdown="Loaded `{}` as `{}` and wrote artifacts to `{}`.".format(path.name, source_kind, artifacts_dir),
    )


def find_local_sample() -> Optional[Path]:
    here = Path(__file__).resolve()
    candidates = [here.parent / "sample_report.json"]
    for parent in (here.parent, *tuple(here.parents)):
        candidates.append(parent / "examples" / "synthetic" / "report.json")
        candidates.append(parent / "examples" / "a100_vllm" / "report.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def coerce_upload_path(uploaded: Any) -> Optional[Path]:
    if uploaded is None:
        return None
    if isinstance(uploaded, (str, Path)):
        path = Path(uploaded)
        return path if path.exists() else None
    if isinstance(uploaded, Sequence) and uploaded:
        first = uploaded[0]
        if isinstance(first, (str, Path)):
            path = Path(first)
            return path if path.exists() else None
    return None
