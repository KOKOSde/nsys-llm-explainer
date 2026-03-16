"""Interactive dashboard for Nsight Systems LLM trace analysis."""

from __future__ import annotations

import argparse
import base64
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots

from .queries import TraceDB, correlate_nvlink_with_nccl, timeline_events
from .report import analyze

HARDWARE_PRESETS: Mapping[str, Mapping[str, float]] = {
    "H100 SXM5": {"bw_tbps": 3.35, "peak_tflops": 989.0},
    "A100 SXM4": {"bw_tbps": 2.0, "peak_tflops": 312.0},
    "A100 PCIe": {"bw_tbps": 1.6, "peak_tflops": 312.0},
}

KERNEL_COLORS: Mapping[str, str] = {
    "compute": "#4ea5ff",
    "memory": "#ff9f43",
    "nccl": "#34c759",
    "other": "#9ea4ad",
}

TIMELINE_COLORS: Mapping[str, str] = {
    "cuda_kernel": "#4ea5ff",
    "memory_kernel": "#ff9f43",
    "memcpy": "#ffd166",
    "nccl": "#34c759",
    "cpu_stall": "#ff4d4f",
    "idle": "#8c8c8c",
}

EXPORT_TEMPLATE = "plotly_dark"

ROOT_STYLE: Mapping[str, Any] = {
    "margin": 0,
    "background": "#0b0f14",
    "color": "#e2e8f0",
    "fontFamily": "Segoe UI, Tahoma, sans-serif",
}

SIDEBAR_STYLE_OPEN: Mapping[str, Any] = {
    "position": "fixed",
    "left": 0,
    "top": 0,
    "bottom": 0,
    "width": "320px",
    "background": "linear-gradient(180deg, #111826 0%, #0d131d 100%)",
    "borderRight": "1px solid #233043",
    "padding": "12px",
    "overflowY": "auto",
    "zIndex": 20,
}

SIDEBAR_STYLE_CLOSED: Mapping[str, Any] = dict(SIDEBAR_STYLE_OPEN, **{"width": "64px"})

MAIN_STYLE_OPEN: Mapping[str, Any] = {"marginLeft": "340px", "padding": "16px 20px"}
MAIN_STYLE_CLOSED: Mapping[str, Any] = {"marginLeft": "84px", "padding": "16px 20px"}

SIDEBAR_LABEL_STYLE: Mapping[str, Any] = {
    "color": "#e2e8f0",
    "fontSize": "13px",
    "fontWeight": 500,
}

SLIDER_MARKS: Mapping[int, Mapping[str, Any]] = {
    value: {"label": str(value), "style": {"color": "#e2e8f0"}}
    for value in range(0, 51, 10)
}


def _coerce_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _coerce_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _classify_kernel_type(name: str) -> str:
    text = str(name or "").lower()
    if "nccl" in text or "allreduce" in text or "allgather" in text or "reducescatter" in text:
        return "nccl"
    if any(token in text for token in ("memcpy", "memset", "copy", "load", "store", "gather", "scatter", "transpose")):
        return "memory"
    if any(token in text for token in ("gemm", "mma", "matmul", "attention", "fused", "conv", "ffn")):
        return "compute"
    return "other"


def _truncate_kernel_label(name: Any, *, index: int, max_chars: int = 64) -> str:
    text = str(name or "unknown")
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return "{:02d}. {}".format(int(index), text)


def _detect_framework(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics") or {}
    corpus: List[str] = []
    for row in (metrics.get("top_kernels") or {}).get("kernels") or []:
        corpus.append(str(row.get("kernel_name") or ""))
    for row in (metrics.get("nvtx") or {}).get("ranges") or []:
        corpus.append(str(row.get("range_name") or ""))
    joined = " ".join(corpus).lower()
    if "vllm" in joined:
        return "vLLM"
    if "sglang" in joined:
        return "SGLang"
    return "other"


def _top_kernel_row(report: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    rows = ((report.get("metrics") or {}).get("top_kernels") or {}).get("kernels") or []
    return rows[0] if rows else None


def _safe_trace_name(report: Mapping[str, Any]) -> str:
    trace_path = ((report.get("trace") or {}).get("path") or report.get("_source_name") or "")
    if trace_path:
        return Path(str(trace_path)).name
    return "unknown"


def _bottleneck_sentence(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics") or {}
    total_gpu_ms = _coerce_float((metrics.get("top_kernels") or {}).get("total_kernel_time_ns")) / 1_000_000.0
    top_kernel = _top_kernel_row(report)
    nccl_ops = (metrics.get("nccl") or {}).get("ops") or []
    top_nccl = nccl_ops[0] if nccl_ops else None
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


def _delta_sentence(current: Mapping[str, Any], baseline: Optional[Mapping[str, Any]]) -> str:
    if not baseline:
        return "No baseline loaded"
    cur_top = _top_kernel_row(current)
    base_top = _top_kernel_row(baseline)
    if not cur_top or not base_top:
        return "Baseline delta unavailable"
    current_name = str(cur_top.get("kernel_name") or "")
    base_rows = ((baseline.get("metrics") or {}).get("top_kernels") or {}).get("kernels") or []
    base_map = {str(row.get("kernel_name") or ""): _coerce_float(row.get("total_time_ms")) for row in base_rows}
    cur_ms = _coerce_float(cur_top.get("total_time_ms"))
    base_ms = base_map.get(current_name, _coerce_float(base_top.get("total_time_ms")))
    if base_ms <= 0:
        return "Baseline delta unavailable"
    delta_pct = ((cur_ms - base_ms) / base_ms) * 100.0
    if delta_pct >= 0:
        return "Top kernel {:.1f}% slower than baseline".format(abs(delta_pct))
    return "Top kernel {:.1f}% faster than baseline".format(abs(delta_pct))


def _decode_upload_content(contents: str) -> bytes:
    header, payload = contents.split(",", 1)
    if "base64" not in header:
        raise ValueError("Upload payload is not base64-encoded.")
    return base64.b64decode(payload)


def _is_sqlite_bytes(blob: bytes) -> bool:
    return blob.startswith(b"SQLite format 3")


def _load_from_sqlite_path(path: Path) -> Dict[str, Any]:
    db = TraceDB.open(path)
    try:
        outputs = analyze(
            db,
            phase_map_path=None,
            kernel_limit=50,
            compute_kernel_percentiles=False,
            compute_nvtx_kernel_map=True,
        )
        report = dict(outputs.report)
        metrics = dict(report.get("metrics") or {})
        timeline = timeline_events(db, limit=50, include_nccl=True)
        metrics["timeline"] = timeline

        # Ensure NVLink output carries timeseries points for panel 4.
        nccl = metrics.get("nccl") or {}
        nvlink = metrics.get("nvlink_during_nccl") or {}
        if "timeseries" not in nvlink:
            nvlink = correlate_nvlink_with_nccl(db, nccl)
        metrics["nvlink_during_nccl"] = nvlink
        report["metrics"] = metrics
        report.setdefault("trace", {})
        report["trace"]["path"] = str(path)
        return _normalize_report(report, source_name=path.name)
    finally:
        db.close()


def _load_from_json_path(path: Path) -> Dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Report JSON root must be an object.")
    report = _normalize_report(parsed, source_name=path.name)
    # Recover NVLink payload from adjacent artifacts/trace when JSON is stale/minimal.
    _hydrate_nvlink_payload(report, report_path=path)
    if not ((report.get("metrics") or {}).get("timeline") or {}).get("events"):
        report["metrics"]["timeline"] = {
            "present": False,
            "events": [],
            "notes": ["Timeline view requires SQLite input. This JSON report has no event-level timeline payload."],
            "total_gpu_time_ms": _coerce_float(((report.get("metrics") or {}).get("top_kernels") or {}).get("total_kernel_time_ns"))
            / 1_000_000.0,
            "total_cpu_time_ms": 0.0,
        }
    return report


def _load_input_path(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError("Input not found: {}".format(path))
    lower = path.suffix.lower()
    if lower in (".sqlite", ".db"):
        return _load_from_sqlite_path(path)
    if lower == ".json":
        return _load_from_json_path(path)
    header = path.read_bytes()[:32]
    if _is_sqlite_bytes(header):
        return _load_from_sqlite_path(path)
    return _load_from_json_path(path)


def _load_from_uploaded_blob(contents: str, filename: Optional[str]) -> Dict[str, Any]:
    blob = _decode_upload_content(contents)
    name = filename or "uploaded"
    suffix = Path(name).suffix.lower()
    if suffix == ".json":
        parsed = json.loads(blob.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("Uploaded JSON root must be an object.")
        return _normalize_report(parsed, source_name=name)
    if suffix in (".sqlite", ".db") or _is_sqlite_bytes(blob):
        with tempfile.NamedTemporaryFile(prefix="nsys_dash_", suffix=".sqlite", delete=False) as tmp:
            tmp.write(blob)
            tmp_path = Path(tmp.name)
        try:
            return _load_from_sqlite_path(tmp_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    parsed = json.loads(blob.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Uploaded file is neither SQLite nor JSON report.")
    return _normalize_report(parsed, source_name=name)


def _normalize_report(report: Mapping[str, Any], *, source_name: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(report)
    metrics: Dict[str, Any] = dict(out.get("metrics") or {})
    out["metrics"] = metrics
    out["_source_name"] = source_name or _safe_trace_name(out)

    metrics.setdefault("top_kernels", {"kernels": [], "total_kernel_time_ns": 0})
    metrics.setdefault("nccl", {"present": False, "ops": [], "windows": [], "pids": []})
    metrics.setdefault("nvlink_during_nccl", {"present": False, "rows": [], "timeseries": [], "notes": []})
    metrics.setdefault("gpu_idle", {"gaps": [], "devices": []})
    metrics.setdefault("barriers", {"barriers": []})
    metrics.setdefault("per_pid", {"pids": []})
    metrics.setdefault("timeline", {"present": False, "events": [], "notes": [], "total_gpu_time_ms": 0.0, "total_cpu_time_ms": 0.0})
    metrics.setdefault("copy_engine", {"present": False, "events": [], "notes": []})
    metrics.setdefault("launch_latency", {"present": False, "summary": {}, "histogram": [], "rows": [], "notes": []})
    metrics.setdefault("stream_overlap", {"present": False, "summary": [], "pairwise": {}, "notes": []})
    metrics.setdefault("phase_split", {"present": False, "rows": [], "source": None, "notes": []})
    metrics.setdefault("roofline", {"present": False, "rows": [], "notes": []})

    nvlink = metrics.get("nvlink_during_nccl") or {}
    if "timeseries" not in nvlink:
        nvlink["timeseries"] = []
    metrics["nvlink_during_nccl"] = nvlink
    return out


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    text = str(x).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _hydrate_nvlink_payload(report: Dict[str, Any], *, report_path: Path) -> None:
    metrics = report.get("metrics") or {}
    nvlink = metrics.get("nvlink_during_nccl") or {}
    if nvlink.get("timeseries"):
        return

    loaded_sidecar = False
    sidecar_ts = report_path.parent / "tables" / "nvlink_timeseries.csv"
    if sidecar_ts.exists() and sidecar_ts.stat().st_size > 0:
        try:
            frame = pd.read_csv(sidecar_ts)
            if not frame.empty:
                rows: List[Dict[str, Any]] = []
                for item in frame.to_dict(orient="records"):
                    rows.append(
                        {
                            "metric_source_id": item.get("metric_source_id"),
                            "metric_name": str(item.get("metric_name") or ""),
                            "timestamp_ns": _coerce_int(item.get("timestamp_ns"), 0),
                            "timestamp_ms": _coerce_float(item.get("timestamp_ms"), 0.0),
                            "metric_value": _coerce_float(item.get("metric_value"), 0.0),
                            "nccl_active": _to_bool(item.get("nccl_active")),
                        }
                    )
                if rows:
                    nvlink["timeseries"] = rows
                    nvlink["present"] = True
                    notes = list(nvlink.get("notes") or [])
                    notes.append("Loaded NVLink timeseries from sidecar tables/nvlink_timeseries.csv.")
                    nvlink["notes"] = notes
                    loaded_sidecar = True
        except Exception:
            loaded_sidecar = False

    if loaded_sidecar:
        metrics["nvlink_during_nccl"] = nvlink
        report["metrics"] = metrics
        return

    trace_path_str = str((report.get("trace") or {}).get("path") or "").strip()
    if not trace_path_str:
        metrics["nvlink_during_nccl"] = nvlink
        report["metrics"] = metrics
        return

    trace_path = Path(trace_path_str)
    if not trace_path.is_absolute():
        trace_path = report_path.parent / trace_path
    if not trace_path.exists():
        metrics["nvlink_during_nccl"] = nvlink
        report["metrics"] = metrics
        return
    if trace_path.suffix.lower() not in {".sqlite", ".db"}:
        metrics["nvlink_during_nccl"] = nvlink
        report["metrics"] = metrics
        return

    try:
        db = TraceDB.open(trace_path)
        try:
            refreshed = correlate_nvlink_with_nccl(db, metrics.get("nccl") or {})
        finally:
            db.close()
        if refreshed.get("timeseries") or refreshed.get("rows"):
            notes = list(refreshed.get("notes") or [])
            notes.append("Recomputed NVLink metrics from trace SQLite during dashboard load.")
            refreshed["notes"] = notes
            nvlink = refreshed
    except Exception:
        pass

    metrics["nvlink_during_nccl"] = nvlink
    report["metrics"] = metrics


def _kernel_df(report: Mapping[str, Any], threshold_ms: float, top_n: int = 20) -> pd.DataFrame:
    rows = ((report.get("metrics") or {}).get("top_kernels") or {}).get("kernels") or []
    if not rows:
        return pd.DataFrame(columns=["kernel_name", "total_ms", "call_count", "avg_us", "pct_total", "kernel_type"])
    data: List[Dict[str, Any]] = []
    for row in rows:
        name = str(row.get("kernel_name") or "unknown")
        total_ms = _coerce_float(row.get("total_time_ms"))
        call_count = _coerce_int(row.get("call_count"))
        avg_us = _coerce_float(row.get("avg_duration_us"))
        pct = _coerce_float(row.get("pct_total_kernel_time"))
        data.append(
            {
                "kernel_name": name,
                "total_ms": total_ms,
                "call_count": call_count,
                "avg_us": avg_us,
                "pct_total": pct,
                "kernel_type": _classify_kernel_type(name),
            }
        )
    df = pd.DataFrame(data).sort_values("total_ms", ascending=False)
    filtered = df[df["total_ms"] >= float(threshold_ms)]
    if filtered.empty:
        filtered = df.head(top_n)
    return filtered.head(top_n)


def _build_kernel_waterfall(
    current: Mapping[str, Any],
    baseline: Optional[Mapping[str, Any]],
    threshold_ms: float,
) -> go.Figure:
    fig = go.Figure()
    cdf = _kernel_df(current, threshold_ms, top_n=20)
    if cdf.empty:
        fig.add_annotation(text="No kernel data found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template=EXPORT_TEMPLATE, title="Kernel Waterfall")
        return fig

    names = list(cdf["kernel_name"])
    display_names = [_truncate_kernel_label(name, index=idx + 1) for idx, name in enumerate(names)]
    colors = [KERNEL_COLORS.get(str(k), KERNEL_COLORS["other"]) for k in cdf["kernel_type"]]
    fig.add_trace(
        go.Bar(
            name="Current",
            orientation="h",
            y=display_names,
            x=cdf["total_ms"],
            marker=dict(color=colors),
            customdata=cdf[["call_count", "avg_us", "pct_total", "kernel_name"]],
            hovertemplate=(
                "Kernel: %{customdata[3]}<br>"
                "Total: %{x:.3f} ms<br>"
                "Calls: %{customdata[0]}<br>"
                "Mean: %{customdata[1]:.2f} us<br>"
                "GPU Time Share: %{customdata[2]:.1f}%<extra></extra>"
            ),
        )
    )

    if baseline:
        bdf = _kernel_df(baseline, threshold_ms, top_n=200)
        base_map = {str(row["kernel_name"]): float(row["total_ms"]) for _, row in bdf.iterrows()}
        bvals = [base_map.get(name, 0.0) for name in names]
        fig.add_trace(
            go.Bar(
                name="Baseline",
                orientation="h",
                y=display_names,
                x=bvals,
                marker=dict(color=colors, pattern=dict(shape="/", fgcolor="#f0f0f0")),
                opacity=0.65,
                customdata=[[name] for name in names],
                hovertemplate="Kernel: %{customdata[0]}<br>Baseline: %{x:.3f} ms<extra></extra>",
            )
        )
        fig.update_layout(barmode="group")

    fig.update_layout(
        template=EXPORT_TEMPLATE,
        title="Kernel Waterfall (Top 20 by Total Duration)",
        xaxis_title="Total Time (ms)",
        yaxis_title="Kernel (truncated)",
        height=520,
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(display_names)),
        tickfont=dict(size=11),
    )
    return fig


def _build_roofline(report: Mapping[str, Any], threshold_ms: float, preset_name: str) -> go.Figure:
    fig = go.Figure()
    roofline = ((report.get("metrics") or {}).get("roofline") or {})
    rows = list(roofline.get("rows") or [])
    preset = HARDWARE_PRESETS.get(preset_name, HARDWARE_PRESETS["H100 SXM5"])
    bw = _coerce_float(preset.get("bw_tbps"), 3.35)
    peak = _coerce_float(preset.get("peak_tflops"), 989.0)

    if not rows:
        kernels = (((report.get("metrics") or {}).get("top_kernels") or {}).get("kernels") or [])
        if not kernels:
            message = "Real roofline counters not found in trace"
            notes = roofline.get("notes") or []
            if notes:
                message = str(notes[0])
            fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(template=EXPORT_TEMPLATE, title="Roofline Scatter")
            return fig

        kdf = pd.DataFrame(kernels)
        kdf["total_time_ms"] = pd.to_numeric(kdf.get("total_time_ms"), errors="coerce").fillna(0.0)
        kdf = kdf[kdf["total_time_ms"] >= float(threshold_ms)]
        if kdf.empty:
            kdf = pd.DataFrame(kernels)
            kdf["total_time_ms"] = pd.to_numeric(kdf.get("total_time_ms"), errors="coerce").fillna(0.0)
        kdf = kdf.sort_values("total_time_ms", ascending=False).head(40)

        max_time_ms = max(1e-6, _coerce_float(kdf["total_time_ms"].max(), 1.0))
        ai_base = {"compute": 24.0, "memory": 1.5, "nccl": 0.25, "other": 4.0}
        eff_base = {"compute": 0.62, "memory": 0.28, "nccl": 0.08, "other": 0.33}

        intensities: List[float] = []
        achieved_tflops: List[float] = []
        sizes: List[float] = []
        colors: List[str] = []
        customdata: List[List[Any]] = []
        color_map = {"memory-bound": "#ff4d4f", "compute-bound": "#34c759"}

        for _, row in kdf.iterrows():
            kernel_name = str(row.get("kernel_name") or "unknown")
            total_time_ms = max(1e-6, _coerce_float(row.get("total_time_ms"), 0.0))
            call_count = int(_coerce_int(row.get("call_count"), 1))
            ktype = _classify_kernel_type(kernel_name)
            weight = max(0.05, min(1.0, total_time_ms / max_time_ms))

            ai = max(1e-6, _coerce_float(ai_base.get(ktype, 4.0)) * (0.7 + (0.8 * weight)))
            tflops = max(1e-6, peak * _coerce_float(eff_base.get(ktype, 0.33)) * (0.55 + (0.6 * weight)))
            tflops = min(tflops, peak * 0.98)

            flops = tflops * 1e12 * (total_time_ms / 1000.0)
            bytes_value = flops / ai if ai > 0 else 0.0
            bound = "memory-bound" if (ai * bw) < tflops else "compute-bound"

            intensities.append(ai)
            achieved_tflops.append(tflops)
            sizes.append(8.0 + math.sqrt(max(1.0, float(call_count))))
            colors.append(color_map.get(bound, "#34c759"))
            customdata.append([kernel_name, call_count, flops, bytes_value])

        fig.add_trace(
            go.Scatter(
                x=intensities,
                y=achieved_tflops,
                mode="markers",
                marker=dict(size=sizes, color=colors, line=dict(width=0.5, color="#111")),
                customdata=customdata,
                hovertemplate=(
                    "Kernel: %{customdata[0]}<br>"
                    "Calls: %{customdata[1]}<br>"
                    "Estimated FLOPs: %{customdata[2]:.3e}<br>"
                    "Estimated Bytes: %{customdata[3]:.3e}<br>"
                    "Arithmetic Intensity: %{x:.3f}<br>"
                    "Estimated Throughput: %{y:.3f} TFLOP/s<extra></extra>"
                ),
                name="Estimated (fallback)",
            )
        )

        min_log = math.log10(max(1e-3, min(intensities) / 2.0))
        max_log = math.log10(max(1.0, max(intensities) * 2.0))
        x_line = [10 ** (min_log + (max_log - min_log) * (i / 100.0)) for i in range(101)]
        mem_line = [bw * x for x in x_line]
        peak_line = [peak for _ in x_line]
        fig.add_trace(go.Scatter(x=x_line, y=mem_line, mode="lines", line=dict(color="#ff9f43", width=2), name="Memory Ceiling"))
        fig.add_trace(go.Scatter(x=x_line, y=peak_line, mode="lines", line=dict(color="#7bd3ff", width=2, dash="dash"), name="Peak Compute"))
        fig.add_annotation(
            text="Using estimated fallback: no real FLOP/byte counters were found in this trace.",
            x=0.01,
            y=0.98,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=11, color="#c9d7ea"),
            bgcolor="rgba(15,22,34,0.75)",
            bordercolor="#233043",
            borderwidth=1,
        )
        fig.update_layout(
            template=EXPORT_TEMPLATE,
            title="Roofline Scatter ({})".format(preset_name),
            xaxis_title="Arithmetic Intensity (estimated)",
            yaxis_title="Throughput (estimated TFLOP/s)",
            xaxis_type="log",
            height=520,
            margin=dict(l=20, r=20, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        )
        return fig

    df = pd.DataFrame(rows)
    df["total_time_ms"] = pd.to_numeric(df["total_time_ms"], errors="coerce").fillna(0.0)
    df = df[df["total_time_ms"] >= float(threshold_ms)]
    if df.empty:
        df = pd.DataFrame(rows)
    df = df.sort_values("total_time_ms", ascending=False).head(40)

    intensities = [max(1e-6, _coerce_float(value)) for value in df["arithmetic_intensity"]]
    achieved_tflops = [max(1e-6, _coerce_float(value)) for value in df["achieved_tflops"]]
    bound_labels = ["memory-bound" if (ai * bw) < peak else "compute-bound" for ai in intensities]
    sizes = [8.0 + math.sqrt(max(1.0, _coerce_float(value))) for value in df["call_count"]]

    color_map = {"memory-bound": "#ff4d4f", "compute-bound": "#34c759"}
    colors = [color_map.get(label, "#34c759") for label in bound_labels]
    fig.add_trace(
        go.Scatter(
            x=intensities,
            y=achieved_tflops,
            mode="markers",
            marker=dict(size=sizes, color=colors, line=dict(width=0.5, color="#111")),
            customdata=df[["kernel_name", "call_count", "flops", "bytes"]],
            hovertemplate=(
                "Kernel: %{customdata[0]}<br>"
                "Calls: %{customdata[1]}<br>"
                "FLOPs: %{customdata[2]:.3e}<br>"
                "Bytes: %{customdata[3]:.3e}<br>"
                "Arithmetic Intensity: %{x:.3f} FLOP/byte<br>"
                "Throughput: %{y:.3f} TFLOP/s<extra></extra>"
            ),
            name="Kernels",
        )
    )

    min_log = math.log10(max(1e-3, min(intensities) / 2.0))
    max_log = math.log10(max(1.0, max(intensities) * 2.0))
    x_line = [10 ** (min_log + (max_log - min_log) * (i / 100.0)) for i in range(101)]
    mem_line = [bw * x for x in x_line]
    peak_line = [peak for _ in x_line]
    fig.add_trace(go.Scatter(x=x_line, y=mem_line, mode="lines", line=dict(color="#ff9f43", width=2), name="Memory Ceiling"))
    fig.add_trace(go.Scatter(x=x_line, y=peak_line, mode="lines", line=dict(color="#7bd3ff", width=2, dash="dash"), name="Peak Compute"))

    fig.update_layout(
        template=EXPORT_TEMPLATE,
        title="Roofline Scatter ({})".format(preset_name),
        xaxis_title="Arithmetic Intensity (FLOP/byte)",
        yaxis_title="Achieved Throughput (TFLOP/s)",
        xaxis_type="log",
        height=520,
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def _build_timeline(report: Mapping[str, Any], include_nccl: bool) -> go.Figure:
    fig = go.Figure()
    timeline = ((report.get("metrics") or {}).get("timeline") or {})
    rows = list(timeline.get("events") or [])
    if not include_nccl:
        rows = [row for row in rows if str(row.get("event_class") or "") != "nccl"]
    rows = sorted(rows, key=lambda row: _coerce_float(row.get("duration_ms")), reverse=True)[:50]

    if not rows:
        note = "No timeline events found."
        notes = timeline.get("notes") or []
        if notes:
            note = str(notes[0])
        fig.add_annotation(text=note, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template=EXPORT_TEMPLATE, title="Timeline")
        return fig

    df = pd.DataFrame(rows)
    lane_order = list(dict.fromkeys(df["lane"].tolist()))
    colors = [TIMELINE_COLORS.get(str(v), "#8c8c8c") for v in df["event_class"]]
    fig.add_trace(
        go.Bar(
            orientation="h",
            y=df["lane"],
            x=df["duration_ms"],
            base=df["start_ms"],
            marker=dict(color=colors),
            customdata=df[["event_name", "start_ms", "duration_ms", "stream_id"]],
            hovertemplate=(
                "Name: %{customdata[0]}<br>"
                "Start: %{customdata[1]:.3f} ms<br>"
                "Duration: %{customdata[2]:.3f} ms<br>"
                "Stream ID: %{customdata[3]}<extra></extra>"
            ),
            name="Events",
        )
    )
    fig.update_layout(
        template=EXPORT_TEMPLATE,
        title="Timeline (Top 50 Events by Duration)",
        xaxis_title="Time from first event (ms)",
        yaxis_title="Lane",
        height=520,
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(lane_order)))
    return fig


def _build_nccl_bar(current: Mapping[str, Any], baseline: Optional[Mapping[str, Any]]) -> go.Figure:
    fig = go.Figure()
    c_ops = (((current.get("metrics") or {}).get("nccl") or {}).get("ops") or [])[:12]
    if not c_ops:
        fig.add_annotation(text="No NCCL activity detected.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template=EXPORT_TEMPLATE, title="NCCL Collectives")
        return fig

    names = [str(op.get("op_name") or "unknown") for op in c_ops]
    c_time = [_coerce_float(op.get("total_time_ms")) for op in c_ops]
    c_overlap = [_coerce_float(op.get("compute_overlap_pct")) for op in c_ops]
    fig.add_trace(
        go.Bar(
            x=names,
            y=c_time,
            marker=dict(color="#34c759"),
            customdata=c_overlap,
            name="Current",
            hovertemplate=(
                "Collective: %{x}<br>"
                "Total NCCL Time: %{y:.3f} ms<br>"
                "Compute Overlap: %{customdata:.1f}%<extra></extra>"
            ),
        )
    )

    if baseline:
        b_ops = (((baseline.get("metrics") or {}).get("nccl") or {}).get("ops") or [])[:40]
        b_map = {str(op.get("op_name") or "unknown"): _coerce_float(op.get("total_time_ms")) for op in b_ops}
        b_vals = [b_map.get(name, 0.0) for name in names]
        fig.add_trace(
            go.Bar(
                x=names,
                y=b_vals,
                marker=dict(color="#34c759", pattern=dict(shape="/", fgcolor="#d8ffe2")),
                opacity=0.65,
                name="Baseline",
                hovertemplate="Collective: %{x}<br>Baseline Time: %{y:.3f} ms<extra></extra>",
            )
        )
        fig.update_layout(barmode="group")

    fig.update_layout(
        template=EXPORT_TEMPLATE,
        title="NCCL Collectives: Total Time and Overlap",
        xaxis_title="Collective",
        yaxis_title="Total Time (ms)",
        height=420,
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def _build_nvlink_line(report: Mapping[str, Any]) -> go.Figure:
    fig = go.Figure()
    nvlink = ((report.get("metrics") or {}).get("nvlink_during_nccl") or {})
    timeseries = nvlink.get("timeseries") or []
    if not timeseries:
        rows = list(nvlink.get("rows") or [])
        if rows:
            df_rows = pd.DataFrame(rows)
            df_rows["metric"] = df_rows.apply(
                lambda row: "{} (src:{})".format(
                    str(row.get("metric_names") or "metric"),
                    str(row.get("metric_source_id")),
                ),
                axis=1,
            )
            fig.add_trace(
                go.Bar(
                    x=df_rows["metric"],
                    y=df_rows["avg_metric_during_nccl"],
                    name="Avg During NCCL",
                    marker=dict(color="#34c759"),
                )
            )
            fig.add_trace(
                go.Bar(
                    x=df_rows["metric"],
                    y=df_rows["avg_metric_outside_nccl"],
                    name="Avg Outside NCCL",
                    marker=dict(color="#9ea4ad"),
                )
            )
            fig.update_layout(
                template=EXPORT_TEMPLATE,
                title="NVLink Summary (No Timeseries Exported)",
                xaxis_title="Metric Source",
                yaxis_title="Metric Value (export units)",
                barmode="group",
                height=420,
                margin=dict(l=20, r=20, t=60, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            )
            return fig

        message = "NVLink data not found in trace"
        notes = list(nvlink.get("notes") or [])
        instructions = list(nvlink.get("capture_instructions") or [])
        if notes:
            message = str(notes[0])
        if instructions:
            message = "{}<br><br>{}".format(message, "<br>".join(str(item) for item in instructions[:3]))
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(color="#d8e6ff", size=12),
            bordercolor="#2b3a52",
            borderwidth=1,
            borderpad=8,
            bgcolor="rgba(17,24,38,0.72)",
        )
        fig.update_layout(template=EXPORT_TEMPLATE, title="NVLink Utilization Over Time")
        return fig

    df = pd.DataFrame(timeseries)
    if "timestamp_ns" in df.columns:
        # Prefer integer nanoseconds to avoid precision loss when large absolute timestamps are used.
        ns = pd.to_numeric(df["timestamp_ns"], errors="coerce")
        df = df.assign(_timestamp_ns=ns).dropna(subset=["_timestamp_ns"]).sort_values("_timestamp_ns")
        if df.empty:
            fig.add_annotation(text="NVLink data not found in trace", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(template=EXPORT_TEMPLATE, title="NVLink Utilization Over Time")
            return fig
        t0_ns = int(df["_timestamp_ns"].min())
        df["t_ms"] = (df["_timestamp_ns"] - float(t0_ns)) / 1_000_000.0
    elif "timestamp_ms" in df.columns:
        ms = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        df = df.assign(_timestamp_ms=ms).dropna(subset=["_timestamp_ms"]).sort_values("_timestamp_ms")
        if df.empty:
            fig.add_annotation(text="NVLink data not found in trace", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
            fig.update_layout(template=EXPORT_TEMPLATE, title="NVLink Utilization Over Time")
            return fig
        t0_ms = float(df["_timestamp_ms"].min())
        df["t_ms"] = df["_timestamp_ms"] - t0_ms
    else:
        fig.add_annotation(text="NVLink data not found in trace", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template=EXPORT_TEMPLATE, title="NVLink Utilization Over Time")
        return fig

    df["metric_value"] = pd.to_numeric(df.get("metric_value"), errors="coerce")
    df = df.dropna(subset=["metric_value", "t_ms"])
    if df.empty:
        fig.add_annotation(text="NVLink data not found in trace", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template=EXPORT_TEMPLATE, title="NVLink Utilization Over Time")
        return fig
    if "nccl_active" not in df.columns:
        df["nccl_active"] = False
    max_t_ms = float(pd.to_numeric(df["t_ms"], errors="coerce").max())
    if not math.isfinite(max_t_ms) or max_t_ms < 0:
        max_t_ms = 0.0
    if max_t_ms < 1.0:
        # Small windows look like all-zeros when shown in ms; switch to us automatically.
        df["t_axis"] = pd.to_numeric(df["t_ms"], errors="coerce") * 1000.0
        xaxis_title = "Time from first NVLink sample (us)"
        hover_time_fmt = "%{x:.2f} us"
    else:
        df["t_axis"] = pd.to_numeric(df["t_ms"], errors="coerce")
        xaxis_title = "Time from first NVLink sample (ms)"
        hover_time_fmt = "%{x:.3f} ms"
    df = df.dropna(subset=["t_axis"])
    if len(df) > 3000:
        step = max(1, len(df) // 3000)
        df = df.iloc[::step, :].reset_index(drop=True)

    grouped = df.groupby(["metric_source_id", "metric_name"], dropna=False)
    for (metric_source_id, metric_name), g in grouped:
        fig.add_trace(
            go.Scatter(
                x=g["t_axis"],
                y=g["metric_value"],
                mode="lines",
                name="{} (src:{})".format(str(metric_name), str(metric_source_id)),
                customdata=g[["nccl_active"]],
                hovertemplate=(
                    "Metric: %{fullData.name}<br>"
                    "Time: " + hover_time_fmt + "<br>"
                    "Value: %{y:.3f}<br>"
                    "NCCL active: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template=EXPORT_TEMPLATE,
        title="NVLink Utilization Over Time",
        xaxis_title=xaxis_title,
        yaxis_title="Metric Value (export units)",
        height=420,
        margin=dict(l=20, r=260, t=60, b=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(15,22,34,0.75)",
            bordercolor="#233043",
            borderwidth=1,
            font=dict(size=10),
        ),
    )
    return fig


def _build_overlap_summary(report: Mapping[str, Any]) -> go.Figure:
    fig = go.Figure()
    overlap = ((report.get("metrics") or {}).get("stream_overlap") or {})
    rows = list(overlap.get("summary") or [])
    if not rows:
        fig.add_annotation(text="No overlap data found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template=EXPORT_TEMPLATE, title="Stream Overlap")
        return fig

    df = pd.DataFrame(rows)
    y_vals = pd.to_numeric(df["total_time_ms"], errors="coerce").fillna(0.0)
    positive = y_vals[y_vals > 0]
    use_log_axis = False
    if not positive.empty:
        min_pos = float(positive.min())
        max_pos = float(positive.max())
        if min_pos > 0 and max_pos / min_pos >= 1000.0:
            use_log_axis = True

    fig.add_trace(
        go.Bar(
            x=df["label"],
            y=y_vals,
            marker=dict(color=["#4ea5ff", "#34c759", "#ffd166"][: len(df)]),
            customdata=df[["overlap_pct"]],
            text=["{:.3f} ms".format(float(v)) for v in y_vals],
            textposition="outside",
            hovertemplate=(
                "Category: %{x}<br>"
                "Total Active Time: %{y:.3f} ms<br>"
                "Overlap Share: %{customdata[0]:.1f}%<extra></extra>"
            ),
            name="Active Time",
        )
    )
    pairwise = overlap.get("pairwise") or {}
    fig.add_annotation(
        text=(
            "Compute∩NCCL: {:.3f} ms | Compute∩Memcpy: {:.3f} ms | NCCL∩Memcpy: {:.3f} ms | Max concurrent GPU ops: {}"
        ).format(
            _coerce_float(pairwise.get("compute_nccl_overlap_ms")),
            _coerce_float(pairwise.get("compute_memcpy_overlap_ms")),
            _coerce_float(pairwise.get("nccl_memcpy_overlap_ms")),
            _coerce_int(pairwise.get("max_concurrent_gpu_ops")),
        ),
        x=0.5,
        y=1.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=11, color="#b8c7da"),
    )
    fig.update_layout(
        template=EXPORT_TEMPLATE,
        title="Stream Concurrency / Overlap",
        xaxis_title="Category",
        yaxis_title="Active Time (ms{})".format(", log scale" if use_log_axis else ""),
        yaxis_type=("log" if use_log_axis else "linear"),
        height=420,
        margin=dict(l=20, r=20, t=80, b=40),
        showlegend=False,
    )
    return fig


def _build_launch_latency(report: Mapping[str, Any]) -> go.Figure:
    fig = go.Figure()
    launch = ((report.get("metrics") or {}).get("launch_latency") or {})
    histogram = list(launch.get("histogram") or [])
    if not histogram:
        note = "No launch latency data found."
        notes = launch.get("notes") or []
        if notes:
            note = str(notes[0])
        fig.add_annotation(text=note, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template=EXPORT_TEMPLATE, title="Launch Latency")
        return fig

    df = pd.DataFrame(histogram)
    centers = (df["bin_start_us"] + df["bin_end_us"]) / 2.0
    widths = (df["bin_end_us"] - df["bin_start_us"]).clip(lower=1e-6)
    fig.add_trace(
        go.Bar(
            x=centers,
            y=df["count"],
            width=widths,
            marker=dict(color="#7bd3ff"),
            hovertemplate=(
                "Latency bin: %{x:.2f} us<br>"
                "Samples: %{y}<extra></extra>"
            ),
            name="Launches",
        )
    )
    summary = launch.get("summary") or {}
    for label, color in (("p50_us", "#34c759"), ("p95_us", "#ff9f43"), ("p99_us", "#ff4d4f")):
        value = summary.get(label)
        if value is None:
            continue
        fig.add_vline(x=float(value), line=dict(color=color, dash="dash"))
    fig.update_layout(
        template=EXPORT_TEMPLATE,
        title=(
            "Launch Latency Distribution "
            "(p50 {:.2f} us | p95 {:.2f} us | p99 {:.2f} us)"
        ).format(
            _coerce_float(summary.get("p50_us")),
            _coerce_float(summary.get("p95_us")),
            _coerce_float(summary.get("p99_us")),
        ),
        xaxis_title="Launch to Kernel Start (us)",
        yaxis_title="Count",
        height=420,
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False,
    )
    return fig


def _build_phase_split(report: Mapping[str, Any]) -> go.Figure:
    fig = go.Figure()
    phases = ((report.get("metrics") or {}).get("phase_split") or {})
    rows = list(phases.get("rows") or [])
    if not rows:
        note = "No phase split data found."
        notes = phases.get("notes") or []
        if notes:
            note = str(notes[0])
        fig.add_annotation(text=note, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template=EXPORT_TEMPLATE, title="Phase Split")
        return fig

    df = pd.DataFrame(rows)
    color_map = {
        "prefill": "#4ea5ff",
        "decode": "#34c759",
        "sampling": "#ff9f43",
        "unclassified": "#8c8c8c",
    }
    colors = [color_map.get(str(name), "#8c8c8c") for name in df["phase"]]
    fig.add_trace(
        go.Bar(
            x=df["phase"],
            y=df["total_time_ms"],
            marker=dict(color=colors),
            customdata=df[["pct_of_total"]],
            hovertemplate=(
                "Phase: %{x}<br>"
                "Time: %{y:.3f} ms<br>"
                "Share: %{customdata[0]:.1f}%<extra></extra>"
            ),
            name="Phase Time",
        )
    )
    fig.update_layout(
        template=EXPORT_TEMPLATE,
        title="Phase Split ({})".format(str(phases.get("source") or "unknown")),
        xaxis_title="Phase",
        yaxis_title="Time (ms)",
        height=420,
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False,
    )
    return fig


def _build_nccl_skew(report: Mapping[str, Any]) -> go.Figure:
    fig = go.Figure()
    rank_rows = list((((report.get("metrics") or {}).get("nccl") or {}).get("rank_rows") or []))
    if not rank_rows:
        fig.add_annotation(text="No per-rank NCCL skew found.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template=EXPORT_TEMPLATE, title="Per-rank NCCL Skew")
        return fig

    df = pd.DataFrame(rank_rows)
    rows = []
    for op_name, group in df.groupby("op_name"):
        top = group.iloc[group["skew_vs_median_pct"].abs().argsort()[::-1]].iloc[0]
        rows.append(
            {
                "op_name": str(op_name),
                "rank_label": str(top["rank_label"]),
                "skew_vs_median_pct": float(top["skew_vs_median_pct"]),
                "total_time_ms": float(top["total_time_ms"]),
            }
        )
    out_df = pd.DataFrame(rows).sort_values("skew_vs_median_pct", key=lambda s: s.abs(), ascending=False).head(16)
    colors = ["#ff4d4f" if value >= 0 else "#34c759" for value in out_df["skew_vs_median_pct"]]
    fig.add_trace(
        go.Bar(
            x=out_df["op_name"],
            y=out_df["skew_vs_median_pct"],
            marker=dict(color=colors),
            customdata=out_df[["rank_label", "total_time_ms"]],
            hovertemplate=(
                "Collective: %{x}<br>"
                "Straggler: %{customdata[0]}<br>"
                "Total Time: %{customdata[1]:.3f} ms<br>"
                "Skew vs median: %{y:.1f}%<extra></extra>"
            ),
            name="Skew",
        )
    )
    fig.update_layout(
        template=EXPORT_TEMPLATE,
        title="Per-rank NCCL Skew",
        xaxis_title="Collective",
        yaxis_title="Skew vs median rank (%)",
        height=420,
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False,
    )
    return fig


def _build_export_figure(
    waterfall: go.Figure,
    roofline: go.Figure,
    timeline: go.Figure,
    nccl: go.Figure,
    nvlink: go.Figure,
    overlap: go.Figure,
    launch_latency: go.Figure,
    phase_split: go.Figure,
    nccl_skew: go.Figure,
) -> go.Figure:
    out = make_subplots(
        rows=5,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}], [{}, {}], [{}, {}]],
        subplot_titles=(
            "Kernel Waterfall",
            "Roofline",
            "Timeline",
            "NCCL Collectives",
            "NVLink Utilization",
            "Stream Overlap",
            "Launch Latency",
            "Phase Split",
            "NCCL Skew",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    for trace in waterfall.data:
        out.add_trace(trace, row=1, col=1)
    for trace in roofline.data:
        out.add_trace(trace, row=2, col=1)
    for trace in timeline.data:
        out.add_trace(trace, row=2, col=2)
    for trace in nccl.data:
        out.add_trace(trace, row=3, col=1)
    for trace in nvlink.data:
        out.add_trace(trace, row=3, col=2)
    for trace in overlap.data:
        out.add_trace(trace, row=4, col=1)
    for trace in launch_latency.data:
        out.add_trace(trace, row=4, col=2)
    for trace in phase_split.data:
        out.add_trace(trace, row=5, col=1)
    for trace in nccl_skew.data:
        out.add_trace(trace, row=5, col=2)
    out.update_layout(template=EXPORT_TEMPLATE, height=2600, title="nsys-llm-explainer Dashboard Export", showlegend=True)
    return out


def _banner_stats(current: Mapping[str, Any], baseline: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    metrics = current.get("metrics") or {}
    timeline = metrics.get("timeline") or {}
    gpu_total_ms = _coerce_float(timeline.get("total_gpu_time_ms"))
    if gpu_total_ms <= 0:
        gpu_total_ms = _coerce_float((metrics.get("top_kernels") or {}).get("total_kernel_time_ns")) / 1_000_000.0
    cpu_total_ms = _coerce_float(timeline.get("total_cpu_time_ms"))
    if cpu_total_ms <= 0:
        sync_rows = (metrics.get("sync") or {}).get("sync_calls") or []
        cpu_total_ms = sum(_coerce_float(row.get("total_time_ms")) for row in sync_rows)

    return {
        "trace": _safe_trace_name(current),
        "gpu": "{:.3f} ms".format(gpu_total_ms),
        "cpu": "{:.3f} ms".format(cpu_total_ms),
        "bottleneck": _bottleneck_sentence(current),
        "framework": _detect_framework(current),
        "delta": _delta_sentence(current, baseline),
    }


def _build_app(initial_report: Dict[str, Any]) -> Dash:
    app = Dash(__name__, title="nsys-llm-explainer Dashboard")
    app.layout = html.Div(
        [
            dcc.Store(id="current-report-store", data=initial_report),
            dcc.Store(id="baseline-report-store", data=None),
            html.Div(
                [
                    html.Button(
                        "Collapse <<",
                        id="sidebar-toggle",
                        n_clicks=0,
                        style={
                            "width": "100%",
                            "marginBottom": "12px",
                            "border": "1px solid #2d3d55",
                            "background": "#152033",
                            "color": "#d7e3f4",
                            "borderRadius": "8px",
                            "padding": "8px",
                            "cursor": "pointer",
                        },
                    ),
                    html.Div(
                        [
                            html.H3("Controls", style={"color": "#f8fbff"}),
                            html.Label("Load trace/report", style=SIDEBAR_LABEL_STYLE),
                            dcc.Upload(
                                id="upload-current",
                                children=html.Div(["Drag and drop .sqlite/.json or click to select"]),
                                style={
                                    "border": "1px dashed #3f5676",
                                    "borderRadius": "10px",
                                    "padding": "12px",
                                    "textAlign": "center",
                                    "background": "#101826",
                                    "cursor": "pointer",
                                },
                                multiple=False,
                            ),
                            html.Div(id="upload-current-status", style={"marginTop": "6px", "fontSize": "12px", "color": "#9fb4cf", "minHeight": "18px"}),
                            html.Br(),
                            html.Label("Compare with baseline", style=SIDEBAR_LABEL_STYLE),
                            dcc.Upload(
                                id="upload-baseline",
                                children=html.Div(["Drag and drop baseline .sqlite/.json"]),
                                style={
                                    "border": "1px dashed #3f5676",
                                    "borderRadius": "10px",
                                    "padding": "12px",
                                    "textAlign": "center",
                                    "background": "#101826",
                                    "cursor": "pointer",
                                },
                                multiple=False,
                            ),
                            html.Div(id="upload-baseline-status", style={"marginTop": "6px", "fontSize": "12px", "color": "#9fb4cf", "minHeight": "18px"}),
                            html.Br(),
                            html.Label("Hardware preset", style=SIDEBAR_LABEL_STYLE),
                            dcc.Dropdown(
                                id="hardware-preset",
                                options=[{"label": key, "value": key} for key in HARDWARE_PRESETS.keys()],
                                value="H100 SXM5",
                                clearable=False,
                            ),
                            html.Br(),
                            html.Label("Show kernels above N ms total time", style=SIDEBAR_LABEL_STYLE),
                            dcc.Slider(
                                id="kernel-threshold",
                                min=0.0,
                                max=50.0,
                                step=0.5,
                                value=1.0,
                                marks=SLIDER_MARKS,
                            ),
                            html.Br(),
                            dcc.Checklist(
                                id="timeline-nccl-toggle",
                                options=[{"label": "Show NCCL events in timeline", "value": "show"}],
                                value=["show"],
                                labelStyle={"color": "#e2e8f0"},
                                inputStyle={"marginRight": "8px"},
                            ),
                            html.Br(),
                            html.Button(
                                "Export Report",
                                id="export-button",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "border": "1px solid #346db2",
                                    "background": "#1f4f86",
                                    "color": "#e7f1ff",
                                    "borderRadius": "8px",
                                    "padding": "9px",
                                    "cursor": "pointer",
                                    "fontWeight": 600,
                                },
                            ),
                            html.Div(id="export-status", style={"marginTop": "6px", "fontSize": "12px", "color": "#9fb4cf", "minHeight": "18px"}),
                        ],
                        id="sidebar-body",
                    ),
                ],
                id="sidebar",
                style=dict(SIDEBAR_STYLE_OPEN),
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [html.Span("Trace file", style={"display": "block", "color": "#8da4c3", "fontSize": "11px", "textTransform": "uppercase"}), html.Div(id="banner-trace", style={"color": "#e7edf6", "fontSize": "14px", "marginTop": "4px"})],
                                style={"border": "1px solid #233043", "borderRadius": "10px", "padding": "10px", "background": "linear-gradient(180deg, #121c2b 0%, #0f1622 100%)", "minHeight": "68px"},
                            ),
                            html.Div(
                                [html.Span("Total GPU time", style={"display": "block", "color": "#8da4c3", "fontSize": "11px", "textTransform": "uppercase"}), html.Div(id="banner-gpu", style={"color": "#e7edf6", "fontSize": "14px", "marginTop": "4px"})],
                                style={"border": "1px solid #233043", "borderRadius": "10px", "padding": "10px", "background": "linear-gradient(180deg, #121c2b 0%, #0f1622 100%)", "minHeight": "68px"},
                            ),
                            html.Div(
                                [html.Span("Total CPU time", style={"display": "block", "color": "#8da4c3", "fontSize": "11px", "textTransform": "uppercase"}), html.Div(id="banner-cpu", style={"color": "#e7edf6", "fontSize": "14px", "marginTop": "4px"})],
                                style={"border": "1px solid #233043", "borderRadius": "10px", "padding": "10px", "background": "linear-gradient(180deg, #121c2b 0%, #0f1622 100%)", "minHeight": "68px"},
                            ),
                            html.Div(
                                [html.Span("Top bottleneck", style={"display": "block", "color": "#8da4c3", "fontSize": "11px", "textTransform": "uppercase"}), html.Div(id="banner-bottleneck", style={"color": "#e7edf6", "fontSize": "14px", "marginTop": "4px", "lineHeight": "1.35"})],
                                style={"border": "1px solid #233043", "borderRadius": "10px", "padding": "10px", "background": "linear-gradient(180deg, #121c2b 0%, #0f1622 100%)", "minHeight": "68px"},
                            ),
                            html.Div(
                                [html.Span("Framework", style={"display": "block", "color": "#8da4c3", "fontSize": "11px", "textTransform": "uppercase"}), html.Div(id="banner-framework", style={"color": "#e7edf6", "fontSize": "14px", "marginTop": "4px"})],
                                style={"border": "1px solid #233043", "borderRadius": "10px", "padding": "10px", "background": "linear-gradient(180deg, #121c2b 0%, #0f1622 100%)", "minHeight": "68px"},
                            ),
                            html.Div(
                                [html.Span("Baseline delta", style={"display": "block", "color": "#8da4c3", "fontSize": "11px", "textTransform": "uppercase"}), html.Div(id="banner-delta", style={"color": "#e7edf6", "fontSize": "14px", "marginTop": "4px"})],
                                style={"border": "1px solid #233043", "borderRadius": "10px", "padding": "10px", "background": "linear-gradient(180deg, #121c2b 0%, #0f1622 100%)", "minHeight": "68px"},
                            ),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(190px, 1fr))", "gap": "10px", "marginBottom": "12px"},
                    ),
                    dcc.Graph(id="kernel-waterfall"),
                    html.Div([dcc.Graph(id="roofline", style={"width": "50%"}), dcc.Graph(id="timeline", style={"width": "50%"})], style={"display": "flex", "gap": "12px", "width": "100%"}),
                    html.Div([dcc.Graph(id="nccl-summary", style={"width": "50%"}), dcc.Graph(id="nvlink-line", style={"width": "50%"})], style={"display": "flex", "gap": "12px", "width": "100%"}),
                    html.Div([dcc.Graph(id="overlap-summary", style={"width": "50%"}), dcc.Graph(id="launch-latency", style={"width": "50%"})], style={"display": "flex", "gap": "12px", "width": "100%"}),
                    html.Div([dcc.Graph(id="phase-split", style={"width": "50%"}), dcc.Graph(id="nccl-skew", style={"width": "50%"})], style={"display": "flex", "gap": "12px", "width": "100%"}),
                ],
                id="main-content",
                style=dict(MAIN_STYLE_OPEN),
            ),
        ],
        style=ROOT_STYLE,
    )

    @app.callback(
        Output("sidebar", "style"),
        Output("sidebar-body", "style"),
        Output("main-content", "style"),
        Output("sidebar-toggle", "children"),
        Input("sidebar-toggle", "n_clicks"),
    )
    def _toggle_sidebar(n_clicks: Optional[int]):
        opened = (int(n_clicks or 0) % 2) == 0
        if opened:
            return (
                dict(SIDEBAR_STYLE_OPEN),
                {"display": "block"},
                dict(MAIN_STYLE_OPEN),
                "Collapse <<",
            )
        return (
            dict(SIDEBAR_STYLE_CLOSED),
            {"display": "none"},
            dict(MAIN_STYLE_CLOSED),
            "Expand >>",
        )

    @app.callback(
        Output("current-report-store", "data"),
        Output("upload-current-status", "children"),
        Input("upload-current", "contents"),
        State("upload-current", "filename"),
        prevent_initial_call=True,
    )
    def _upload_current(contents: Optional[str], filename: Optional[str]):
        if not contents:
            return no_update, "No file selected."
        try:
            report = _load_from_uploaded_blob(contents, filename)
            status = "Loaded {}".format(filename or "uploaded file")
            return report, status
        except Exception as exc:
            return no_update, "Load failed: {}".format(exc)

    @app.callback(
        Output("baseline-report-store", "data"),
        Output("upload-baseline-status", "children"),
        Input("upload-baseline", "contents"),
        State("upload-baseline", "filename"),
        prevent_initial_call=True,
    )
    def _upload_baseline(contents: Optional[str], filename: Optional[str]):
        if not contents:
            return None, "No baseline selected."
        try:
            report = _load_from_uploaded_blob(contents, filename)
            status = "Loaded baseline {}".format(filename or "uploaded file")
            return report, status
        except Exception as exc:
            return no_update, "Baseline load failed: {}".format(exc)

    @app.callback(
        Output("banner-trace", "children"),
        Output("banner-gpu", "children"),
        Output("banner-cpu", "children"),
        Output("banner-bottleneck", "children"),
        Output("banner-framework", "children"),
        Output("banner-delta", "children"),
        Output("kernel-waterfall", "figure"),
        Output("roofline", "figure"),
        Output("timeline", "figure"),
        Output("nccl-summary", "figure"),
        Output("nvlink-line", "figure"),
        Output("overlap-summary", "figure"),
        Output("launch-latency", "figure"),
        Output("phase-split", "figure"),
        Output("nccl-skew", "figure"),
        Input("current-report-store", "data"),
        Input("baseline-report-store", "data"),
        Input("hardware-preset", "value"),
        Input("kernel-threshold", "value"),
        Input("timeline-nccl-toggle", "value"),
    )
    def _render_panels(
        current_data: Mapping[str, Any],
        baseline_data: Optional[Mapping[str, Any]],
        hardware_preset: str,
        threshold_ms: float,
        nccl_toggle: Sequence[str],
    ):
        def _error_fig(title: str, message: str) -> go.Figure:
            fig = go.Figure()
            fig.add_annotation(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                font=dict(color="#d8e6ff", size=12),
                bordercolor="#2b3a52",
                borderwidth=1,
                borderpad=8,
                bgcolor="rgba(17,24,38,0.72)",
            )
            fig.update_layout(template=EXPORT_TEMPLATE, title=title, height=420)
            return fig

        try:
            current = _normalize_report(current_data or {}, source_name="current")
            baseline = _normalize_report(baseline_data, source_name="baseline") if baseline_data else None
            show_nccl = "show" in (nccl_toggle or [])

            stats = _banner_stats(current, baseline)
            waterfall = _build_kernel_waterfall(current, baseline, threshold_ms=float(threshold_ms or 0.0))
            roofline = _build_roofline(current, threshold_ms=float(threshold_ms or 0.0), preset_name=str(hardware_preset or "H100 SXM5"))
            timeline = _build_timeline(current, include_nccl=bool(show_nccl))
            nccl = _build_nccl_bar(current, baseline)
            nvlink = _build_nvlink_line(current)
            overlap = _build_overlap_summary(current)
            launch_latency = _build_launch_latency(current)
            phases = _build_phase_split(current)
            nccl_skew = _build_nccl_skew(current)
            return (
                stats["trace"],
                stats["gpu"],
                stats["cpu"],
                stats["bottleneck"],
                stats["framework"],
                stats["delta"],
                waterfall,
                roofline,
                timeline,
                nccl,
                nvlink,
                overlap,
                launch_latency,
                phases,
                nccl_skew,
            )
        except Exception as exc:
            message = "Dashboard render error: {}".format(str(exc))
            return (
                "load_error",
                "-",
                "-",
                message,
                "unknown",
                "No baseline loaded",
                _error_fig("Kernel Waterfall", message),
                _error_fig("Roofline Scatter", message),
                _error_fig("Timeline", message),
                _error_fig("NCCL Collectives", message),
                _error_fig("NVLink Utilization Over Time", message),
                _error_fig("Stream Overlap", message),
                _error_fig("Launch Latency", message),
                _error_fig("Phase Split", message),
                _error_fig("NCCL Skew", message),
            )

    @app.callback(
        Output("export-status", "children"),
        Input("export-button", "n_clicks"),
        State("current-report-store", "data"),
        State("baseline-report-store", "data"),
        State("hardware-preset", "value"),
        State("kernel-threshold", "value"),
        State("timeline-nccl-toggle", "value"),
        prevent_initial_call=True,
    )
    def _export_dashboard(
        n_clicks: int,
        current_data: Mapping[str, Any],
        baseline_data: Optional[Mapping[str, Any]],
        hardware_preset: str,
        threshold_ms: float,
        nccl_toggle: Sequence[str],
    ) -> str:
        del n_clicks
        current = _normalize_report(current_data or {}, source_name="current")
        baseline = _normalize_report(baseline_data, source_name="baseline") if baseline_data else None
        show_nccl = "show" in (nccl_toggle or [])

        waterfall = _build_kernel_waterfall(current, baseline, threshold_ms=float(threshold_ms or 0.0))
        roofline = _build_roofline(current, threshold_ms=float(threshold_ms or 0.0), preset_name=str(hardware_preset or "H100 SXM5"))
        timeline = _build_timeline(current, include_nccl=bool(show_nccl))
        nccl = _build_nccl_bar(current, baseline)
        nvlink = _build_nvlink_line(current)
        overlap = _build_overlap_summary(current)
        launch_latency = _build_launch_latency(current)
        phases = _build_phase_split(current)
        nccl_skew = _build_nccl_skew(current)
        export_fig = _build_export_figure(
            waterfall,
            roofline,
            timeline,
            nccl,
            nvlink,
            overlap,
            launch_latency,
            phases,
            nccl_skew,
        )

        export_path = Path.cwd() / "dashboard_export_{}.html".format(time.strftime("%Y%m%d_%H%M%S"))
        pio.write_html(export_fig, file=str(export_path), include_plotlyjs="cdn", full_html=True, auto_open=False)
        return "Exported to {}".format(export_path)

    return app


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nsys_llm_explainer.dashboard",
        description="Interactive Plotly Dash dashboard for Nsight Systems LLM traces.",
    )
    p.add_argument("--db", required=True, help="Path to trace.sqlite or report.json.")
    p.add_argument("--host", default="127.0.0.1", help="Dash host (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=8050, help="Dash port (default: 8050).")
    p.add_argument("--debug", action="store_true", help="Run Dash in debug mode.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    initial_report = _load_input_path(str(args.db))
    app = _build_app(initial_report)
    app.run(host=str(args.host), port=int(args.port), debug=bool(args.debug))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
