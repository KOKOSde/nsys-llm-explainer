"""Trace-derived queries/metrics from an Nsight Systems SQLite export."""

import contextlib
import csv
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .schema import SchemaProbeResult, decode_global_pid, decode_global_tid, probe_schema, sqlite_version


@dataclass
class TraceDB:
    path: Path
    conn: sqlite3.Connection
    schema: SchemaProbeResult

    @classmethod
    def open(cls, path: Union[str, Path]) -> "TraceDB":
        p = Path(path)
        conn = sqlite3.connect(str(p))
        conn.row_factory = sqlite3.Row
        schema = probe_schema(conn)
        return cls(path=p, conn=conn, schema=schema)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.conn.close()


def _ns_to_us(ns: Union[int, float]) -> float:
    return float(ns) / 1_000.0


def _ns_to_ms(ns: Union[int, float]) -> float:
    return float(ns) / 1_000_000.0


def _safe_div(n: float, d: float) -> float:
    return (n / d) if d else 0.0


def _fetch_one(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return None if row is None else row[0]


def schema_discovery(trace_db: TraceDB) -> Dict[str, Any]:
    tables: Dict[str, Any] = {}
    for name, info in trace_db.schema.tables.items():
        tables[name] = {"columns": list(info.columns), "types": dict(info.column_types)}
    gpu_metrics_table = _pick_table_with_required_cols(
        trace_db,
        candidates=("GPU_METRICS",),
        required_cols=("timestamp", "metricId", "value"),
        name_hint="GPU_METRICS",
    )
    target_info_gpu_metrics_table = _pick_table_with_required_cols(
        trace_db,
        candidates=("TARGET_INFO_GPU_METRICS",),
        required_cols=("metricId", "metricName"),
        name_hint="GPU_METRICS",
    )
    cuda_graph_table = _pick_table_with_required_cols(
        trace_db,
        candidates=("CUDA_GRAPH_EVENTS",),
        required_cols=("start",),
        name_hint="CUDA_GRAPH",
    )
    kernel_pid_source: Optional[str] = None
    if trace_db.schema.kernel_table:
        kinfo = trace_db.schema.table(trace_db.schema.kernel_table)
        if kinfo.has("pid"):
            kernel_pid_source = "pid"
        elif kinfo.has("processId"):
            kernel_pid_source = "processId"
        elif kinfo.has("globalPid"):
            kernel_pid_source = "globalPid"

    runtime_pid_source: Optional[str] = None
    if trace_db.schema.runtime_table:
        rinfo = trace_db.schema.table(trace_db.schema.runtime_table)
        if rinfo.has("pid"):
            runtime_pid_source = "pid"
        elif rinfo.has("processId"):
            runtime_pid_source = "processId"
        elif rinfo.has("globalTid"):
            runtime_pid_source = "globalTid"
        elif rinfo.has("globalPid"):
            runtime_pid_source = "globalPid"

    nvtx_pid_source: Optional[str] = None
    if trace_db.schema.nvtx_table:
        ninfo = trace_db.schema.table(trace_db.schema.nvtx_table)
        if ninfo.has("pid"):
            nvtx_pid_source = "pid"
        elif ninfo.has("processId"):
            nvtx_pid_source = "processId"
        elif ninfo.has("globalTid"):
            nvtx_pid_source = "globalTid"

    capabilities: Dict[str, Any] = {
        "has_string_table": bool(trace_db.schema.string_table),
        "kernel_table": {
            "present": bool(trace_db.schema.kernel_table),
            "has_deviceId": bool(trace_db.schema.kernel_table and trace_db.schema.table(trace_db.schema.kernel_table).has("deviceId")),
            "has_globalPid": bool(trace_db.schema.kernel_table and trace_db.schema.table(trace_db.schema.kernel_table).has("globalPid")),
            "has_pid": bool(trace_db.schema.kernel_table and trace_db.schema.table(trace_db.schema.kernel_table).has("pid")),
            "has_processId": bool(trace_db.schema.kernel_table and trace_db.schema.table(trace_db.schema.kernel_table).has("processId")),
            "has_correlationId": bool(trace_db.schema.kernel_table and trace_db.schema.table(trace_db.schema.kernel_table).has("correlationId")),
        },
        "runtime_table": {
            "present": bool(trace_db.schema.runtime_table),
            "has_nameId": bool(trace_db.schema.runtime_table and trace_db.schema.table(trace_db.schema.runtime_table).has("nameId")),
            "has_name": bool(trace_db.schema.runtime_table and trace_db.schema.table(trace_db.schema.runtime_table).has("name")),
            "has_globalTid": bool(trace_db.schema.runtime_table and trace_db.schema.table(trace_db.schema.runtime_table).has("globalTid")),
            "has_pid": bool(trace_db.schema.runtime_table and trace_db.schema.table(trace_db.schema.runtime_table).has("pid")),
            "has_processId": bool(trace_db.schema.runtime_table and trace_db.schema.table(trace_db.schema.runtime_table).has("processId")),
            "has_correlationId": bool(trace_db.schema.runtime_table and trace_db.schema.table(trace_db.schema.runtime_table).has("correlationId")),
        },
        "nvtx_table": {
            "present": bool(trace_db.schema.nvtx_table),
            "has_end": bool(trace_db.schema.nvtx_table and trace_db.schema.table(trace_db.schema.nvtx_table).has("end")),
            "has_text": bool(trace_db.schema.nvtx_table and trace_db.schema.table(trace_db.schema.nvtx_table).has("text")),
            "has_textId": bool(trace_db.schema.nvtx_table and trace_db.schema.table(trace_db.schema.nvtx_table).has("textId")),
            "has_globalTid": bool(trace_db.schema.nvtx_table and trace_db.schema.table(trace_db.schema.nvtx_table).has("globalTid")),
        },
        "gpu_metrics_table": {
            "present": bool(gpu_metrics_table),
            "table": gpu_metrics_table,
            "target_info_table": target_info_gpu_metrics_table,
        },
        "cuda_graph_table": {
            "present": bool(cuda_graph_table),
            "table": cuda_graph_table,
        },
    }

    # Timestamp units: Nsight Systems exports `start/end` in nanoseconds for CUDA/CUPTI activity.
    # We still run a small sanity check on the overall kernel time window to flag obviously
    # suspicious unit scales (best-effort; cannot be proven from values alone).
    time_unit_assumed = "ns"
    time_unit_guess = "unknown"
    time_unit_guess_basis: Optional[str] = None
    if trace_db.schema.kernel_table:
        row = trace_db.conn.execute(
            "SELECT MIN(start) AS s, MAX(end) AS e FROM {t}".format(t=trace_db.schema.kernel_table)
        ).fetchone()
        if row and row["s"] is not None and row["e"] is not None:
            window = int(row["e"]) - int(row["s"])
            # If the window is large enough, it is consistent with nanosecond units.
            # (If it were microseconds, these thresholds would imply multi-day traces.)
            if window >= 1_000_000_000:  # >= 1s if ns; >= ~11.6 days if us
                time_unit_guess = "ns"
                time_unit_guess_basis = "kernel_window_ns_ge_1s"
            elif window >= 1_000_000:  # >= 1ms if ns; >= ~16.7 minutes if us
                time_unit_guess = "ns_likely"
                time_unit_guess_basis = "kernel_window_ns_ge_1ms"

    return {
        "sqlite_version": sqlite_version(trace_db.conn),
        "path": str(trace_db.path),
        "string_table": trace_db.schema.string_table,
        "kernel_table": trace_db.schema.kernel_table,
        "runtime_table": trace_db.schema.runtime_table,
        "nvtx_table": trace_db.schema.nvtx_table,
        "gpu_metrics_table": gpu_metrics_table,
        "target_info_gpu_metrics_table": target_info_gpu_metrics_table,
        "cuda_graph_table": cuda_graph_table,
        "sync_table": trace_db.schema.sync_table,
        "kernel_pid_source": kernel_pid_source,
        "runtime_pid_source": runtime_pid_source,
        "nvtx_pid_source": nvtx_pid_source,
        "timestamp_unit_assumed": time_unit_assumed,
        "timestamp_unit_guess": time_unit_guess,
        "timestamp_unit_guess_basis": time_unit_guess_basis,
        "capabilities": capabilities,
        "tables": tables,
    }


def _percentile_from_sorted(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0.0:
        return float(values[0])
    if q >= 1.0:
        return float(values[-1])
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    w = pos - lo
    return float(values[lo]) * (1.0 - w) + float(values[hi]) * w


_NCCL_COLLECTIVE_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("allreduce", ("allreduce", "all_reduce", "all reduce")),
    ("allgather", ("allgather", "all_gather", "all gather")),
    ("reducescatter", ("reducescatter", "reduce_scatter", "reduce scatter")),
    ("broadcast", ("broadcast",)),
)

_NCCL_SIGNAL_PATTERNS: Tuple[str, ...] = (
    "nccl",
    "allreduce",
    "all_reduce",
    "all reduce",
    "allgather",
    "all_gather",
    "all gather",
    "reducescatter",
    "reduce_scatter",
    "reduce scatter",
    "broadcast",
)

_NVLINK_SIGNAL_PATTERNS: Tuple[str, ...] = (
    "nvlink",
    "nvlrx__",
    "nvltx__",
    "nvlrx",
    "nvltx",
)

_SYNC_BARRIER_PATTERNS: Tuple[str, ...] = (
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaStreamWaitEvent",
    "cudaEventQuery",
    "cuCtxSynchronize",
    "cuStreamSynchronize",
    "cuEventSynchronize",
    "cuStreamWaitEvent",
)

_BLOCKING_COPY_PATTERNS: Tuple[str, ...] = (
    "cudaMemcpy",
    "cudaMemcpy2D",
    "cudaMemcpy3D",
    "cudaMemcpyPeer",
    "cuMemcpy",
)

_HOST_WAIT_PATTERNS: Tuple[str, ...] = (
    "poll",
    "ppoll",
    "epoll_wait",
    "select",
    "pselect",
    "futex",
    "pthread_cond_wait",
    "pthread_cond_timedwait",
    "sem_wait",
    "nanosleep",
    "clock_nanosleep",
)

_LAUNCH_API_PATTERNS: Tuple[str, ...] = (
    "cudaLaunch",
    "cudaLaunchKernel",
    "cudaLaunchKernelEx",
    "cudaGraphLaunch",
    "cuLaunchKernel",
    "cuGraphLaunch",
)

_MEMORY_KERNEL_PATTERNS: Tuple[str, ...] = (
    "memcpy",
    "memset",
    "copy",
    "load",
    "store",
    "gather",
    "scatter",
    "transpose",
)


def _lower_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _name_matches_patterns(name: Any, patterns: Sequence[str]) -> bool:
    text = _lower_text(name)
    return any(pattern.lower() in text for pattern in patterns)


def _classify_nccl_collective(*texts: Any) -> Optional[str]:
    joined = " ".join(_lower_text(text) for text in texts if text)
    for label, patterns in _NCCL_COLLECTIVE_PATTERNS:
        if any(pattern in joined for pattern in patterns):
            return label
    return None


def _looks_like_nccl(*texts: Any) -> bool:
    joined = " ".join(_lower_text(text) for text in texts if text)
    if not joined:
        return False
    if "nccl" in joined:
        return True
    collective = _classify_nccl_collective(joined)
    return collective is not None


def _extract_rank_label(*texts: Any) -> Optional[str]:
    joined = " ".join(str(text or "") for text in texts if text)
    if not joined:
        return None
    patterns = (
        r'"rank"\s*:\s*(\d+)',
        r"\brank\s*[:=]\s*(\d+)\b",
        r"\blocal_rank\s*[:=]\s*(\d+)\b",
        r"\bglobal_rank\s*[:=]\s*(\d+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, joined, flags=re.IGNORECASE)
        if match:
            return "rank:{}".format(int(match.group(1)))
    return None


def _interval_overlap_ns(intervals: Sequence[Tuple[int, int]], start_ns: int, end_ns: int) -> int:
    overlap = 0
    for iv_start, iv_end in intervals:
        if iv_end <= start_ns:
            continue
        if iv_start >= end_ns:
            break
        overlap += max(0, min(end_ns, iv_end) - max(start_ns, iv_start))
    return overlap


def _metric_capture_instructions(include_nccl: bool = False) -> List[str]:
    trace_switches = "cuda,nvtx,osrt"
    if include_nccl:
        trace_switches = "nccl,cuda,nvtx,osrt"
    return [
        "NVLink counters not found in the SQLite export.",
        "List supported metric sets first: `nsys profile --gpu-metrics-devices=all --gpu-metrics-set=help`.",
        (
            "Then re-capture with GPU Metrics enabled, for example: "
            "`sudo nsys profile --trace={trace} --cuda-trace-scope=process-tree "
            "--gpu-metrics-devices=all --gpu-metrics-set=<supported-set> "
            "--gpu-metrics-frequency=10000 --cuda-graph-trace=node -o trace <app>`."
        ).format(trace=trace_switches),
        "Export again with SQLite output: `nsys export --type sqlite --output trace.sqlite --force-overwrite=true --lazy=false trace.nsys-rep`.",
    ]


def _resolve_name_expr(
    trace_db: TraceDB,
    *,
    table: str,
    alias: str,
    info: Any,
    text_col: str,
    text_id_col: str,
    string_alias: str,
) -> Tuple[str, str]:
    stable = trace_db.schema.string_table
    join = ""
    if info.has(text_col):
        expr = "{alias}.{col}".format(alias=alias, col=text_col)
        if info.has(text_id_col) and stable and not trace_db.schema.is_text_column(table, text_id_col):
            join = " LEFT JOIN {stable} {s_alias} ON {s_alias}.id = {alias}.{id_col} ".format(
                stable=stable, s_alias=string_alias, alias=alias, id_col=text_id_col
            )
            expr = "COALESCE({alias}.{col}, {s_alias}.value)".format(alias=alias, col=text_col, s_alias=string_alias)
        return (expr, join)
    if info.has(text_id_col):
        if stable and not trace_db.schema.is_text_column(table, text_id_col):
            join = " LEFT JOIN {stable} {s_alias} ON {s_alias}.id = {alias}.{id_col} ".format(
                stable=stable, s_alias=string_alias, alias=alias, id_col=text_id_col
            )
            return ("{s_alias}.value".format(s_alias=string_alias), join)
        return ("{alias}.{id_col}".format(alias=alias, id_col=text_id_col), join)
    return ("''", join)


def _pick_table_with_required_cols(
    trace_db: TraceDB,
    *,
    candidates: Sequence[str],
    required_cols: Sequence[str],
    name_hint: Optional[str] = None,
) -> Optional[str]:
    tables = trace_db.schema.tables
    for candidate in candidates:
        info = tables.get(candidate)
        if info and all(info.has(col) for col in required_cols):
            return candidate
    for name, info in tables.items():
        if name_hint and name_hint.lower() not in name.lower():
            continue
        if all(info.has(col) for col in required_cols):
            return name
    return None


def _classify_barrier_kind(name: str, *, source: str) -> Optional[str]:
    lowered = _lower_text(name)
    if source == "runtime":
        if any(pattern.lower() in lowered for pattern in _BLOCKING_COPY_PATTERNS) and "async" not in lowered:
            return "blocking_memcpy"
        if any(pattern.lower() in lowered for pattern in _SYNC_BARRIER_PATTERNS) or "synchronize" in lowered or "wait" in lowered:
            return "sync_api"
        return None
    if source == "osrt":
        if any(pattern in lowered for pattern in _HOST_WAIT_PATTERNS):
            return "host_wait"
    return None


def get_top_kernels(
    trace_db: TraceDB,
    *,
    limit: int = 30,
    compute_percentiles: bool = True,
    percentiles: Tuple[float, float] = (0.50, 0.90),
) -> Dict[str, Any]:
    ktable = trace_db.schema.kernel_table
    stable = trace_db.schema.string_table
    if not ktable:
        return {
            "table": None,
            "kernels": [],
            "total_kernel_time_ns": 0,
            "notes": ["No CUDA kernel activity table found (expected CUPTI_ACTIVITY_KIND_KERNEL)."],
            "sql": {},
        }

    kinfo = trace_db.schema.table(ktable)
    name_col = "demangledName" if kinfo.has("demangledName") else ("shortName" if kinfo.has("shortName") else None)
    if not name_col:
        return {
            "table": ktable,
            "kernels": [],
            "total_kernel_time_ns": 0,
            "notes": ["Kernel table is missing demangledName/shortName columns."],
            "sql": {},
        }

    has_device = kinfo.has("deviceId")

    sql_total = "SELECT SUM(end - start) FROM {t}".format(t=ktable)
    total_kernel_time_ns = int(_fetch_one(trace_db.conn, sql_total) or 0)

    join = ""
    group_name_expr = "k.{c}".format(c=name_col)
    if stable and not trace_db.schema.is_text_column(ktable, name_col):
        join = " JOIN {s} s ON s.id = k.{c} ".format(s=stable, c=name_col)
        group_name_expr = "s.value"

    device_expr = "k.deviceId" if has_device else "-1"
    sql_agg = (
        "SELECT {name} AS kernel_name, {dev} AS device_id, "
        "COUNT(*) AS call_count, "
        "SUM(k.end - k.start) AS total_time_ns, "
        "AVG(k.end - k.start) AS avg_time_ns, "
        "MIN(k.end - k.start) AS min_time_ns, "
        "MAX(k.end - k.start) AS max_time_ns "
        "FROM {t} k {join} "
        "GROUP BY kernel_name, device_id "
        "ORDER BY total_time_ns DESC "
        "LIMIT ?"
    ).format(name=group_name_expr, dev=device_expr, t=ktable, join=join)

    rows = trace_db.conn.execute(sql_agg, (int(limit),)).fetchall()
    kernels: List[Dict[str, Any]] = []
    for r in rows:
        total_ns = int(r["total_time_ns"] or 0)
        kernels.append(
            {
                "kernel_name": str(r["kernel_name"]),
                "device_id": int(r["device_id"]) if r["device_id"] is not None else None,
                "call_count": int(r["call_count"] or 0),
                "total_time_ns": total_ns,
                "total_time_ms": _ns_to_ms(total_ns),
                "avg_duration_us": _ns_to_us(float(r["avg_time_ns"] or 0.0)),
                "min_duration_us": _ns_to_us(float(r["min_time_ns"] or 0.0)),
                "max_duration_us": _ns_to_us(float(r["max_time_ns"] or 0.0)),
                "p50_duration_us": None,
                "p90_duration_us": None,
                "pct_total_kernel_time": (_safe_div(float(total_ns), float(total_kernel_time_ns)) * 100.0)
                if total_kernel_time_ns
                else 0.0,
            }
        )

    notes: List[str] = []
    if stable is None and (not trace_db.schema.is_text_column(ktable, name_col)):
        notes.append("String table not found; kernel_name values may be numeric string IDs.")
    sql: Dict[str, str] = {"agg": sql_agg, "total": sql_total}

    if compute_percentiles and kernels:
        p50, p90 = percentiles
        for k in kernels:
            kname = k["kernel_name"]
            dev = k["device_id"]
            where = ["1=1"]
            params: List[Any] = []
            if stable and not trace_db.schema.is_text_column(ktable, name_col):
                where.append("s.value = ?")
                params.append(kname)
                join2 = " JOIN {s} s ON s.id = k.{c} ".format(s=stable, c=name_col)
            else:
                where.append("k.{c} = ?".format(c=name_col))
                params.append(kname)
                join2 = ""
            if has_device:
                where.append("k.deviceId = ?")
                params.append(int(dev) if dev is not None else -1)
            sql_durs = "SELECT (k.end - k.start) AS dur_ns FROM {t} k {join} WHERE {w} ORDER BY dur_ns".format(
                t=ktable, join=join2, w=" AND ".join(where)
            )
            durs = [float(rr[0]) for rr in trace_db.conn.execute(sql_durs, params).fetchall()]
            if not durs:
                continue
            k["p50_duration_us"] = _ns_to_us(_percentile_from_sorted(durs, p50) or 0.0)
            k["p90_duration_us"] = _ns_to_us(_percentile_from_sorted(durs, p90) or 0.0)
        sql["durations"] = "SELECT (end-start) FROM {t} ... ORDER BY".format(t=ktable)
    elif not compute_percentiles:
        notes.append("Kernel percentiles skipped (compute_percentiles=False).")

    return {"table": ktable, "kernels": kernels, "total_kernel_time_ns": total_kernel_time_ns, "notes": notes, "sql": sql}


def _kernel_events_basic(trace_db: TraceDB) -> List[Dict[str, Any]]:
    ktable = trace_db.schema.kernel_table
    if not ktable:
        return []
    kinfo = trace_db.schema.table(ktable)
    cols = ["start", "end"]
    if kinfo.has("deviceId"):
        cols.append("deviceId")
    sql = "SELECT {cols} FROM {t} ORDER BY start".format(cols=", ".join(cols), t=ktable)
    rows = trace_db.conn.execute(sql).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        start = int(r["start"] or 0)
        end = int(r["end"] or 0)
        out.append(
            {
                "start_ns": start,
                "end_ns": end,
                "dur_ns": max(0, end - start),
                "device_id": int(r["deviceId"]) if ("deviceId" in r.keys() and r["deviceId"] is not None) else None,
            }
        )
    return out


def detect_launch_storm(
    kernels: Optional[Mapping[str, Any]],
    *,
    trace_db: Optional[TraceDB] = None,
    tiny_kernel_us: float = 5.0,
    tiny_kernel_limit: int = 10,
) -> Dict[str, Any]:
    if trace_db is None:
        return {
            "total_launches": 0,
            "window_s": 0.0,
            "launches_per_s": 0.0,
            "median_kernel_us": None,
            "tiny_kernel_us": float(tiny_kernel_us),
            "tiny_kernels": [],
            "notes": ["No trace_db provided; launch storm requires raw kernel activity."],
            "sql": {},
        }

    events = _kernel_events_basic(trace_db)
    if not events:
        return {
            "total_launches": 0,
            "window_s": 0.0,
            "launches_per_s": 0.0,
            "median_kernel_us": None,
            "tiny_kernel_us": float(tiny_kernel_us),
            "tiny_kernels": [],
            "notes": ["No kernel events found."],
            "sql": {},
        }

    starts = [e["start_ns"] for e in events]
    ends = [e["end_ns"] for e in events]
    window_start = min(starts)
    window_end = max(ends)
    window_ns = max(1, window_end - window_start)
    window_s = float(window_ns) / 1_000_000_000.0

    durs_us = sorted([_ns_to_us(e["dur_ns"]) for e in events])
    median_us = _percentile_from_sorted(durs_us, 0.50)
    p50_us = _percentile_from_sorted(durs_us, 0.50)
    p90_us = _percentile_from_sorted(durs_us, 0.90)
    p99_us = _percentile_from_sorted(durs_us, 0.99)

    n = float(len(durs_us))
    pct_under_5us = (_safe_div(float(sum(1 for d in durs_us if d < 5.0)), n) * 100.0) if n else 0.0
    pct_under_10us = (_safe_div(float(sum(1 for d in durs_us if d < 10.0)), n) * 100.0) if n else 0.0
    pct_under_20us = (_safe_div(float(sum(1 for d in durs_us if d < 20.0)), n) * 100.0) if n else 0.0

    # Classification thresholds live in heuristics.py to make them easy to tune.
    try:
        from .heuristics import LAUNCH_STORM_THRESHOLDS, classify_launch_storm
    except Exception:
        LAUNCH_STORM_THRESHOLDS = {}
        classify_launch_storm = None

    launches_per_s = _safe_div(float(len(events)), window_s)
    is_storm = bool(classify_launch_storm(launches_per_s, float(p50_us or 0.0))) if classify_launch_storm else None

    ktable = trace_db.schema.kernel_table
    stable = trace_db.schema.string_table
    tiny: List[Dict[str, Any]] = []
    sql_tiny = None
    if ktable:
        kinfo = trace_db.schema.table(ktable)
        name_col = "demangledName" if kinfo.has("demangledName") else ("shortName" if kinfo.has("shortName") else None)
        if name_col:
            join = ""
            name_expr = "k.{c}".format(c=name_col)
            if stable and not trace_db.schema.is_text_column(ktable, name_col):
                join = " JOIN {s} s ON s.id = k.{c} ".format(s=stable, c=name_col)
                name_expr = "s.value"
            sql_tiny = (
                "SELECT {name} AS kernel_name, COUNT(*) AS call_count, AVG(k.end-k.start) AS avg_dur_ns "
                "FROM {t} k {join} "
                "WHERE (k.end-k.start) <= ? "
                "GROUP BY kernel_name "
                "ORDER BY call_count DESC "
                "LIMIT ?"
            ).format(name=name_expr, t=ktable, join=join)
            rows = trace_db.conn.execute(sql_tiny, (int(tiny_kernel_us * 1000.0), int(tiny_kernel_limit))).fetchall()
            for r in rows:
                tiny.append(
                    {
                        "kernel_name": str(r["kernel_name"]),
                        "call_count": int(r["call_count"] or 0),
                        "avg_duration_us": _ns_to_us(float(r["avg_dur_ns"] or 0.0)),
                    }
                )

    return {
        "total_launches": len(events),
        "window_s": window_s,
        "launches_per_s": launches_per_s,
        "median_kernel_us": float(median_us) if median_us is not None else None,
        "p50_kernel_us": float(p50_us) if p50_us is not None else None,
        "p90_kernel_us": float(p90_us) if p90_us is not None else None,
        "p99_kernel_us": float(p99_us) if p99_us is not None else None,
        "pct_under_5us": float(pct_under_5us),
        "pct_under_10us": float(pct_under_10us),
        "pct_under_20us": float(pct_under_20us),
        "is_launch_storm": is_storm,
        "storm_thresholds": LAUNCH_STORM_THRESHOLDS,
        "tiny_kernel_us": float(tiny_kernel_us),
        "tiny_kernels": tiny,
        "notes": [],
        "sql": {"tiny_kernels": sql_tiny} if sql_tiny else {},
    }


def find_sync_events(trace_db: TraceDB, *, limit: int = 200) -> Dict[str, Any]:
    rtable = trace_db.schema.runtime_table
    stable = trace_db.schema.string_table
    if not rtable:
        return {"table": None, "sync_calls": [], "notes": ["No runtime API activity table found."], "sql": {}}

    rinfo = trace_db.schema.table(rtable)
    name_col = "nameId" if rinfo.has("nameId") else ("name" if rinfo.has("name") else None)
    if not name_col:
        return {"table": rtable, "sync_calls": [], "notes": ["Runtime table missing name/nameId."], "sql": {}}

    join = ""
    name_expr = "r.{c}".format(c=name_col)
    if stable and not trace_db.schema.is_text_column(rtable, name_col):
        join = " JOIN {s} s ON s.id = r.{c} ".format(s=stable, c=name_col)
        name_expr = "s.value"

    sync_keywords = [
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cudaEventSynchronize",
        "cudaStreamWaitEvent",
        "cudaEventQuery",
        "cuCtxSynchronize",
        "cuStreamSynchronize",
        "cuEventSynchronize",
        "cuStreamWaitEvent",
    ]
    where_parts = ["({expr} LIKE ?)".format(expr=name_expr) for _ in range(len(sync_keywords) + 2)]
    params: List[Any] = ["%{}%".format(k) for k in sync_keywords] + ["%Wait%", "%Synchronize%"]

    sql = (
        "SELECT {name} AS api_name, COUNT(*) AS call_count, "
        "SUM(r.end - r.start) AS total_time_ns, AVG(r.end-r.start) AS avg_time_ns "
        "FROM {t} r {join} "
        "WHERE " + " OR ".join(where_parts) + " "
        "GROUP BY api_name "
        "ORDER BY total_time_ns DESC "
        "LIMIT ?"
    ).format(name=name_expr, t=rtable, join=join)
    params.append(int(limit))

    rows = trace_db.conn.execute(sql, params).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        total_ns = int(r["total_time_ns"] or 0)
        out.append(
            {
                "api_name": str(r["api_name"]),
                "call_count": int(r["call_count"] or 0),
                "total_time_ms": _ns_to_ms(total_ns),
                "avg_duration_us": _ns_to_us(float(r["avg_time_ns"] or 0.0)),
            }
        )

    notes: List[str] = []
    if stable is None and (not trace_db.schema.is_text_column(rtable, name_col)):
        notes.append("String table not found; api_name values may be numeric string IDs.")
    return {"table": rtable, "sync_calls": out, "notes": notes, "sql": {"sync_calls": sql}}


def find_cpu_gpu_barriers(
    trace_db: TraceDB,
    *,
    limit: int = 50,
    launcher_gap_threshold_us: float = 50.0,
) -> Dict[str, Any]:
    barrier_rows: List[Dict[str, Any]] = []
    barrier_rows_by_pid: List[Dict[str, Any]] = []
    per_pid_totals: Dict[int, Dict[str, Any]] = {}
    notes: List[str] = []
    sql: Dict[str, str] = {}

    def _update_pid_summary(pid: Optional[int], total_ns: int, api_name: str, barrier_kind: str, count: int) -> None:
        if pid is None:
            return
        bucket = per_pid_totals.setdefault(
            int(pid),
            {
                "pid": int(pid),
                "total_barrier_time_ns": 0,
                "barrier_event_count": 0,
                "top_barrier": None,
                "top_barrier_kind": None,
                "top_barrier_time_ns": 0,
            },
        )
        bucket["total_barrier_time_ns"] += int(total_ns)
        bucket["barrier_event_count"] += int(count)
        if int(total_ns) > int(bucket["top_barrier_time_ns"]):
            bucket["top_barrier"] = str(api_name)
            bucket["top_barrier_kind"] = str(barrier_kind)
            bucket["top_barrier_time_ns"] = int(total_ns)

    runtime_barriers: Dict[Tuple[str, str], Dict[str, Any]] = {}
    runtime_barriers_by_pid: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    launch_rows: List[Dict[str, Any]] = []

    rtable = trace_db.schema.runtime_table
    stable = trace_db.schema.string_table
    if rtable:
        rinfo = trace_db.schema.table(rtable)
        name_col = "nameId" if rinfo.has("nameId") else ("name" if rinfo.has("name") else None)
        pid_expr, _pid_source, _ = _pid_expr_for_table("r", rinfo)
        if name_col:
            name_expr = "r.{c}".format(c=name_col)
            join = ""
            if stable and not trace_db.schema.is_text_column(rtable, name_col):
                join = " JOIN {s} s ON s.id = r.{c} ".format(s=stable, c=name_col)
                name_expr = "s.value"

            filter_patterns = sorted(set(_SYNC_BARRIER_PATTERNS + _BLOCKING_COPY_PATTERNS + _LAUNCH_API_PATTERNS + ("Wait", "Synchronize")))
            where_parts = ["LOWER({expr}) LIKE ?".format(expr=name_expr) for _ in filter_patterns]
            params: List[Any] = ["%{}%".format(pattern.lower()) for pattern in filter_patterns]
            pid_select = "{pid} AS pid,".format(pid=pid_expr) if pid_expr else "NULL AS pid,"
            sql_runtime = (
                "SELECT {pid_select} {name} AS api_name, r.start AS start_ns, r.end AS end_ns "
                "FROM {table} r {join} "
                "WHERE r.end IS NOT NULL AND r.end > r.start AND (" + " OR ".join(where_parts) + ") "
                "ORDER BY pid, start_ns"
            ).format(pid_select=pid_select, name=name_expr, table=rtable, join=join)
            sql["runtime_barriers"] = sql_runtime
            for row in trace_db.conn.execute(sql_runtime, tuple(params)).fetchall():
                api_name = str(row["api_name"])
                total_ns = max(0, int(row["end_ns"] or 0) - int(row["start_ns"] or 0))
                pid = int(row["pid"]) if row["pid"] is not None else None
                barrier_kind = _classify_barrier_kind(api_name, source="runtime")
                if barrier_kind:
                    key = (barrier_kind, api_name)
                    bucket = runtime_barriers.setdefault(
                        key,
                        {
                            "barrier_kind": barrier_kind,
                            "api_name": api_name,
                            "count": 0,
                            "total_time_ns": 0,
                            "max_duration_ns": 0,
                        },
                    )
                    bucket["count"] += 1
                    bucket["total_time_ns"] += total_ns
                    bucket["max_duration_ns"] = max(int(bucket["max_duration_ns"]), int(total_ns))
                    if pid is not None:
                        pid_key = (pid, barrier_kind, api_name)
                        pid_bucket = runtime_barriers_by_pid.setdefault(
                            pid_key,
                            {
                                "pid": pid,
                                "barrier_kind": barrier_kind,
                                "api_name": api_name,
                                "count": 0,
                                "total_time_ns": 0,
                                "max_duration_ns": 0,
                            },
                        )
                        pid_bucket["count"] += 1
                        pid_bucket["total_time_ns"] += total_ns
                        pid_bucket["max_duration_ns"] = max(int(pid_bucket["max_duration_ns"]), int(total_ns))
                if _name_matches_patterns(api_name, _LAUNCH_API_PATTERNS):
                    launch_rows.append(
                        {
                            "pid": pid,
                            "api_name": api_name,
                            "start_ns": int(row["start_ns"] or 0),
                            "end_ns": int(row["end_ns"] or 0),
                        }
                    )
        else:
            notes.append("Runtime table present but missing name/nameId; barrier detection limited.")
    else:
        notes.append("No runtime API table found; barrier detection limited to OS runtime waits if available.")

    osrt_table = _pick_table_with_required_cols(
        trace_db,
        candidates=("OSRT_API",),
        required_cols=("start", "end"),
        name_hint="OSRT",
    )
    if osrt_table:
        oinfo = trace_db.schema.table(osrt_table)
        name_col = "nameId" if oinfo.has("nameId") else ("name" if oinfo.has("name") else None)
        pid_expr, _pid_source, _ = _pid_expr_for_table("o", oinfo)
        if name_col:
            name_expr = "o.{c}".format(c=name_col)
            join = ""
            if stable and not trace_db.schema.is_text_column(osrt_table, name_col):
                join = " JOIN {s} s ON s.id = o.{c} ".format(s=stable, c=name_col)
                name_expr = "s.value"
            where_parts = ["LOWER({expr}) LIKE ?".format(expr=name_expr) for _ in _HOST_WAIT_PATTERNS]
            params = ["%{}%".format(pattern.lower()) for pattern in _HOST_WAIT_PATTERNS]
            pid_select = "{pid} AS pid,".format(pid=pid_expr) if pid_expr else "NULL AS pid,"
            sql_osrt = (
                "SELECT {pid_select} {name} AS api_name, COUNT(*) AS call_count, "
                "SUM(o.end-o.start) AS total_time_ns, AVG(o.end-o.start) AS avg_time_ns, "
                "MAX(o.end-o.start) AS max_time_ns "
                "FROM {table} o {join} "
                "WHERE o.end IS NOT NULL AND o.end > o.start AND (" + " OR ".join(where_parts) + ") "
                "GROUP BY pid, api_name "
                "ORDER BY total_time_ns DESC"
            ).format(pid_select=pid_select, name=name_expr, table=osrt_table, join=join)
            sql["osrt_waits"] = sql_osrt
            for row in trace_db.conn.execute(sql_osrt, tuple(params)).fetchall():
                api_name = str(row["api_name"])
                pid = int(row["pid"]) if row["pid"] is not None else None
                total_ns = int(row["total_time_ns"] or 0)
                barrier_rows.append(
                    {
                        "barrier_kind": "host_wait",
                        "api_name": api_name,
                        "count": int(row["call_count"] or 0),
                        "total_time_ms": _ns_to_ms(total_ns),
                        "avg_duration_us": _ns_to_us(float(row["avg_time_ns"] or 0.0)),
                        "max_duration_us": _ns_to_us(float(row["max_time_ns"] or 0.0)),
                    }
                )
                if pid is not None:
                    barrier_rows_by_pid.append(
                        {
                            "pid": pid,
                            "barrier_kind": "host_wait",
                            "api_name": api_name,
                            "count": int(row["call_count"] or 0),
                            "total_time_ms": _ns_to_ms(total_ns),
                            "avg_duration_us": _ns_to_us(float(row["avg_time_ns"] or 0.0)),
                            "max_duration_us": _ns_to_us(float(row["max_time_ns"] or 0.0)),
                        }
                    )
                    _update_pid_summary(pid, total_ns, api_name, "host_wait", int(row["call_count"] or 0))

    for row in runtime_barriers.values():
        total_ns = int(row["total_time_ns"] or 0)
        barrier_rows.append(
            {
                "barrier_kind": str(row["barrier_kind"]),
                "api_name": str(row["api_name"]),
                "count": int(row["count"] or 0),
                "total_time_ms": _ns_to_ms(total_ns),
                "avg_duration_us": _ns_to_us(_safe_div(float(total_ns), float(row["count"] or 0))),
                "max_duration_us": _ns_to_us(float(row["max_duration_ns"] or 0.0)),
            }
        )

    for row in runtime_barriers_by_pid.values():
        total_ns = int(row["total_time_ns"] or 0)
        pid = int(row["pid"])
        barrier_rows_by_pid.append(
            {
                "pid": pid,
                "barrier_kind": str(row["barrier_kind"]),
                "api_name": str(row["api_name"]),
                "count": int(row["count"] or 0),
                "total_time_ms": _ns_to_ms(total_ns),
                "avg_duration_us": _ns_to_us(_safe_div(float(total_ns), float(row["count"] or 0))),
                "max_duration_us": _ns_to_us(float(row["max_duration_ns"] or 0.0)),
            }
        )
        _update_pid_summary(pid, total_ns, str(row["api_name"]), str(row["barrier_kind"]), int(row["count"] or 0))

    launcher_gap_threshold_ns = int(float(launcher_gap_threshold_us) * 1_000.0)
    if launch_rows:
        launch_rows.sort(key=lambda item: ((item["pid"] if item["pid"] is not None else -1), item["start_ns"], item["end_ns"]))
        gaps_global: List[int] = []
        gaps_by_pid: Dict[int, List[int]] = {}
        previous_end_by_pid: Dict[Optional[int], int] = {}
        for row in launch_rows:
            pid = row["pid"]
            previous_end = previous_end_by_pid.get(pid)
            if previous_end is not None:
                gap_ns = max(0, int(row["start_ns"]) - int(previous_end))
                if gap_ns >= launcher_gap_threshold_ns:
                    gaps_global.append(gap_ns)
                    if pid is not None:
                        gaps_by_pid.setdefault(int(pid), []).append(gap_ns)
            previous_end_by_pid[pid] = max(int(previous_end or 0), int(row["end_ns"]))

        if gaps_global:
            gaps_global.sort()
            total_ns = sum(gaps_global)
            barrier_rows.append(
                {
                    "barrier_kind": "cpu_launcher_gap",
                    "api_name": "cpu_launcher_gap",
                    "count": len(gaps_global),
                    "total_time_ms": _ns_to_ms(total_ns),
                    "avg_duration_us": _ns_to_us(_safe_div(float(total_ns), float(len(gaps_global)))),
                    "max_duration_us": _ns_to_us(float(gaps_global[-1])),
                }
            )
            for pid, gaps in gaps_by_pid.items():
                gaps.sort()
                total_pid_ns = sum(gaps)
                barrier_rows_by_pid.append(
                    {
                        "pid": int(pid),
                        "barrier_kind": "cpu_launcher_gap",
                        "api_name": "cpu_launcher_gap",
                        "count": len(gaps),
                        "total_time_ms": _ns_to_ms(total_pid_ns),
                        "avg_duration_us": _ns_to_us(_safe_div(float(total_pid_ns), float(len(gaps)))),
                        "max_duration_us": _ns_to_us(float(gaps[-1])),
                    }
                )
                _update_pid_summary(int(pid), int(total_pid_ns), "cpu_launcher_gap", "cpu_launcher_gap", len(gaps))

    barrier_rows.sort(key=lambda item: (-float(item.get("total_time_ms") or 0.0), -int(item.get("count") or 0), str(item.get("api_name") or "")))
    barrier_rows_by_pid.sort(
        key=lambda item: (
            int(item.get("pid") or -1),
            -float(item.get("total_time_ms") or 0.0),
            -int(item.get("count") or 0),
            str(item.get("api_name") or ""),
        )
    )

    pid_summaries: List[Dict[str, Any]] = []
    for pid, row in sorted(per_pid_totals.items(), key=lambda item: item[1]["total_barrier_time_ns"], reverse=True):
        pid_summaries.append(
            {
                "pid": int(pid),
                "total_barrier_time_ms": _ns_to_ms(int(row["total_barrier_time_ns"] or 0)),
                "barrier_event_count": int(row["barrier_event_count"] or 0),
                "top_barrier": row.get("top_barrier"),
                "top_barrier_kind": row.get("top_barrier_kind"),
            }
        )

    return {
        "present": bool(barrier_rows),
        "barriers": barrier_rows[: int(limit)],
        "barriers_by_pid": barrier_rows_by_pid,
        "pids": pid_summaries,
        "launcher_gap_threshold_us": float(launcher_gap_threshold_us),
        "notes": notes,
        "sql": sql,
    }


def detect_nccl_ops(trace_db: TraceDB, *, limit: int = 20) -> Dict[str, Any]:
    selected_events: List[Dict[str, Any]] = []
    notes: List[str] = []
    sql: Dict[str, str] = {}

    def _collect_nvtx_events() -> List[Dict[str, Any]]:
        ntable = trace_db.schema.nvtx_table
        if not ntable:
            return []
        ninfo = trace_db.schema.table(ntable)
        if not ninfo.has("end"):
            return []

        pid_expr, _pid_source, _ = _pid_expr_for_table("n", ninfo)
        range_expr, join = _resolve_name_expr(
            trace_db,
            table=ntable,
            alias="n",
            info=ninfo,
            text_col="text",
            text_id_col="textId",
            string_alias="sn",
        )
        payload_expr, payload_join = _resolve_name_expr(
            trace_db,
            table=ntable,
            alias="n",
            info=ninfo,
            text_col="jsonText",
            text_id_col="jsonTextId",
            string_alias="sj",
        )
        full_join = (join or "") + (payload_join or "")
        predicates: List[str] = []
        params: List[Any] = []
        for pattern in _NCCL_SIGNAL_PATTERNS:
            predicates.append("LOWER(COALESCE({expr}, '')) LIKE ?".format(expr=range_expr))
            params.append("%{}%".format(pattern.lower()))
            if payload_expr != "''":
                predicates.append("LOWER(COALESCE({expr}, '')) LIKE ?".format(expr=payload_expr))
                params.append("%{}%".format(pattern.lower()))
        pid_select = "{pid} AS pid,".format(pid=pid_expr) if pid_expr else "NULL AS pid,"
        event_type_filter = ""
        if ninfo.has("eventType"):
            event_type_filter = " AND n.eventType IN (59, 60)"
        sql_events = (
            "SELECT {pid_select} {name} AS range_name, {payload} AS payload_json, "
            "n.start AS start_ns, n.end AS end_ns "
            "FROM {table} n {join} "
            "WHERE n.end IS NOT NULL AND n.end > n.start{event_type_filter} AND (" + " OR ".join(predicates) + ") "
            "ORDER BY n.start"
        ).format(
            pid_select=pid_select,
            name=range_expr,
            payload=payload_expr,
            table=ntable,
            join=full_join,
            event_type_filter=event_type_filter,
        )
        sql["nccl_nvtx"] = sql_events
        rows: List[Dict[str, Any]] = []
        for row in trace_db.conn.execute(sql_events, tuple(params)).fetchall():
            range_name = str(row["range_name"] or "")
            payload_json = str(row["payload_json"] or "")
            if not _looks_like_nccl(range_name, payload_json):
                continue
            rows.append(
                {
                    "source": "nvtx",
                    "raw_name": range_name or "nccl_op",
                    "op_name": _classify_nccl_collective(range_name, payload_json) or range_name or "nccl_op",
                    "start_ns": int(row["start_ns"] or 0),
                    "end_ns": int(row["end_ns"] or 0),
                    "dur_ns": max(0, int(row["end_ns"] or 0) - int(row["start_ns"] or 0)),
                    "pid": int(row["pid"]) if row["pid"] is not None else None,
                    "rank_label": _extract_rank_label(range_name, payload_json),
                    "device_id": None,
                }
            )
        return rows

    def _collect_runtime_events() -> List[Dict[str, Any]]:
        rtable = trace_db.schema.runtime_table
        if not rtable:
            return []
        rinfo = trace_db.schema.table(rtable)
        name_col = "nameId" if rinfo.has("nameId") else ("name" if rinfo.has("name") else None)
        if not name_col:
            return []
        pid_expr, _pid_source, _ = _pid_expr_for_table("r", rinfo)
        name_expr = "r.{c}".format(c=name_col)
        join = ""
        stable = trace_db.schema.string_table
        if stable and not trace_db.schema.is_text_column(rtable, name_col):
            join = " JOIN {s} s ON s.id = r.{c} ".format(s=stable, c=name_col)
            name_expr = "s.value"
        predicates = ["LOWER({expr}) LIKE ?".format(expr=name_expr) for _ in _NCCL_SIGNAL_PATTERNS]
        params = ["%{}%".format(pattern.lower()) for pattern in _NCCL_SIGNAL_PATTERNS]
        pid_select = "{pid} AS pid,".format(pid=pid_expr) if pid_expr else "NULL AS pid,"
        sql_events = (
            "SELECT {pid_select} {name} AS api_name, r.start AS start_ns, r.end AS end_ns "
            "FROM {table} r {join} "
            "WHERE r.end IS NOT NULL AND r.end > r.start AND (" + " OR ".join(predicates) + ") "
            "ORDER BY r.start"
        ).format(pid_select=pid_select, name=name_expr, table=rtable, join=join)
        sql["nccl_runtime"] = sql_events
        rows: List[Dict[str, Any]] = []
        for row in trace_db.conn.execute(sql_events, tuple(params)).fetchall():
            api_name = str(row["api_name"] or "")
            if not _looks_like_nccl(api_name):
                continue
            rows.append(
                {
                    "source": "runtime",
                    "raw_name": api_name or "nccl_op",
                    "op_name": _classify_nccl_collective(api_name) or api_name or "nccl_op",
                    "start_ns": int(row["start_ns"] or 0),
                    "end_ns": int(row["end_ns"] or 0),
                    "dur_ns": max(0, int(row["end_ns"] or 0) - int(row["start_ns"] or 0)),
                    "pid": int(row["pid"]) if row["pid"] is not None else None,
                    "rank_label": _extract_rank_label(api_name),
                    "device_id": None,
                }
            )
        return rows

    def _collect_kernel_events() -> List[Dict[str, Any]]:
        ktable = trace_db.schema.kernel_table
        if not ktable:
            return []
        kinfo = trace_db.schema.table(ktable)
        name_col = "demangledName" if kinfo.has("demangledName") else ("shortName" if kinfo.has("shortName") else None)
        if not name_col:
            return []
        pid_expr, _pid_source, _ = _pid_expr_for_table("k", kinfo)
        name_expr = "k.{c}".format(c=name_col)
        join = ""
        stable = trace_db.schema.string_table
        if stable and not trace_db.schema.is_text_column(ktable, name_col):
            join = " JOIN {s} s ON s.id = k.{c} ".format(s=stable, c=name_col)
            name_expr = "s.value"
        predicates = ["LOWER({expr}) LIKE ?".format(expr=name_expr)]
        params = ["%nccl%"]
        pid_select = "{pid} AS pid,".format(pid=pid_expr) if pid_expr else "NULL AS pid,"
        dev_select = "k.deviceId AS device_id," if kinfo.has("deviceId") else "NULL AS device_id,"
        sql_events = (
            "SELECT {pid_select} {dev_select} {name} AS kernel_name, k.start AS start_ns, k.end AS end_ns "
            "FROM {table} k {join} "
            "WHERE k.end IS NOT NULL AND k.end > k.start AND (" + " OR ".join(predicates) + ") "
            "ORDER BY k.start"
        ).format(pid_select=pid_select, dev_select=dev_select, name=name_expr, table=ktable, join=join)
        sql["nccl_kernels"] = sql_events
        rows: List[Dict[str, Any]] = []
        for row in trace_db.conn.execute(sql_events, tuple(params)).fetchall():
            kernel_name = str(row["kernel_name"] or "")
            if not _looks_like_nccl(kernel_name):
                continue
            rows.append(
                {
                    "source": "kernel",
                    "raw_name": kernel_name or "nccl_op",
                    "op_name": _classify_nccl_collective(kernel_name) or kernel_name or "nccl_op",
                    "start_ns": int(row["start_ns"] or 0),
                    "end_ns": int(row["end_ns"] or 0),
                    "dur_ns": max(0, int(row["end_ns"] or 0) - int(row["start_ns"] or 0)),
                    "pid": int(row["pid"]) if row["pid"] is not None else None,
                    "rank_label": _extract_rank_label(kernel_name),
                    "device_id": int(row["device_id"]) if row["device_id"] is not None else None,
                }
            )
        return rows

    nvtx_events = _collect_nvtx_events()
    runtime_events = _collect_runtime_events()
    kernel_events = _collect_kernel_events()

    if nvtx_events:
        selected_events = nvtx_events
        notes.append("Using NVTX ranges as NCCL windows (best-effort; works with Nsight Systems NCCL trace or app-emitted NCCL NVTX labels).")
    elif runtime_events:
        selected_events = runtime_events
        notes.append("Using runtime API calls as NCCL windows.")
    elif kernel_events:
        selected_events = kernel_events
        notes.append("Using NCCL kernel names as NCCL windows; collective names may be inferred only from kernel names.")
    else:
        return {
            "present": False,
            "source": None,
            "ops": [],
            "pids": [],
            "event_count": 0,
            "windows": [],
            "notes": ["No NCCL-like activity found in NVTX ranges, runtime API rows, or kernel names."],
            "sql": sql,
        }

    compute_intervals: List[Tuple[int, int]] = []
    ktable = trace_db.schema.kernel_table
    if ktable:
        kinfo = trace_db.schema.table(ktable)
        name_col = "demangledName" if kinfo.has("demangledName") else ("shortName" if kinfo.has("shortName") else None)
        if name_col:
            name_expr = "k.{c}".format(c=name_col)
            join = ""
            stable = trace_db.schema.string_table
            if stable and not trace_db.schema.is_text_column(ktable, name_col):
                join = " JOIN {s} s ON s.id = k.{c} ".format(s=stable, c=name_col)
                name_expr = "s.value"
            sql_compute = (
                "SELECT k.start AS start_ns, k.end AS end_ns, {name} AS kernel_name "
                "FROM {table} k {join} "
                "WHERE k.end IS NOT NULL AND k.end > k.start "
                "ORDER BY k.start"
            ).format(name=name_expr, table=ktable, join=join)
            sql["compute_overlap"] = sql_compute
            for row in trace_db.conn.execute(sql_compute).fetchall():
                kernel_name = str(row["kernel_name"] or "")
                if _looks_like_nccl(kernel_name):
                    continue
                compute_intervals.append((int(row["start_ns"] or 0), int(row["end_ns"] or 0)))
    merged_compute_intervals = _merge_intervals(compute_intervals)

    op_buckets: Dict[str, Dict[str, Any]] = {}
    pid_buckets: Dict[int, Dict[str, Any]] = {}
    windows = _merge_intervals([(int(event["start_ns"]), int(event["end_ns"])) for event in selected_events])
    for event in selected_events:
        overlap_ns = _interval_overlap_ns(merged_compute_intervals, int(event["start_ns"]), int(event["end_ns"]))
        event["compute_overlap_ns"] = overlap_ns
        label = str(event["op_name"])
        bucket = op_buckets.setdefault(
            label,
            {
                "op_name": label,
                "raw_name_example": str(event["raw_name"]),
                "count": 0,
                "total_time_ns": 0,
                "max_duration_ns": 0,
                "total_compute_overlap_ns": 0,
                "source": str(event["source"]),
                "straggler_label": None,
                "straggler_total_ns": 0,
                "straggler_max_ns": 0,
                "_stragglers": {},
            },
        )
        bucket["count"] += 1
        bucket["total_time_ns"] += int(event["dur_ns"])
        bucket["max_duration_ns"] = max(int(bucket["max_duration_ns"]), int(event["dur_ns"]))
        bucket["total_compute_overlap_ns"] += int(overlap_ns)
        straggler_label = event.get("rank_label")
        if straggler_label is None and event.get("pid") is not None:
            straggler_label = "pid:{}".format(int(event["pid"]))
        if straggler_label is not None:
            sb = bucket["_stragglers"].setdefault(straggler_label, {"total_ns": 0, "max_ns": 0})
            sb["total_ns"] += int(event["dur_ns"])
            sb["max_ns"] = max(int(sb["max_ns"]), int(event["dur_ns"]))
        if event.get("pid") is not None:
            pid = int(event["pid"])
            pid_bucket = pid_buckets.setdefault(
                pid,
                {
                    "pid": pid,
                    "total_nccl_time_ns": 0,
                    "nccl_event_count": 0,
                    "max_duration_ns": 0,
                    "top_nccl_op": None,
                    "top_nccl_op_time_ns": 0,
                },
            )
            pid_bucket["total_nccl_time_ns"] += int(event["dur_ns"])
            pid_bucket["nccl_event_count"] += 1
            pid_bucket["max_duration_ns"] = max(int(pid_bucket["max_duration_ns"]), int(event["dur_ns"]))
            op_total = pid_bucket.get("_ops", {})
            op_total[label] = op_total.get(label, 0) + int(event["dur_ns"])
            pid_bucket["_ops"] = op_total

    ops: List[Dict[str, Any]] = []
    for bucket in op_buckets.values():
        best_label = None
        best_total_ns = -1
        best_max_ns = -1
        for straggler_label, values in bucket["_stragglers"].items():
            total_ns = int(values["total_ns"])
            max_ns = int(values["max_ns"])
            if (total_ns, max_ns, str(straggler_label)) > (best_total_ns, best_max_ns, str(best_label)):
                best_label = straggler_label
                best_total_ns = total_ns
                best_max_ns = max_ns
        total_ns = int(bucket["total_time_ns"] or 0)
        overlap_ns = int(bucket["total_compute_overlap_ns"] or 0)
        ops.append(
            {
                "op_name": str(bucket["op_name"]),
                "raw_name_example": str(bucket["raw_name_example"]),
                "source": str(bucket["source"]),
                "count": int(bucket["count"] or 0),
                "total_time_ms": _ns_to_ms(total_ns),
                "max_duration_ms": _ns_to_ms(int(bucket["max_duration_ns"] or 0)),
                "avg_duration_us": _ns_to_us(_safe_div(float(total_ns), float(bucket["count"] or 0))),
                "compute_overlap_ms": _ns_to_ms(overlap_ns),
                "compute_overlap_pct": (_safe_div(float(overlap_ns), float(total_ns)) * 100.0) if total_ns else 0.0,
                "straggler": best_label,
                "straggler_total_ms": _ns_to_ms(best_total_ns) if best_total_ns >= 0 else None,
                "straggler_max_ms": _ns_to_ms(best_max_ns) if best_max_ns >= 0 else None,
            }
        )
    ops.sort(key=lambda item: (-float(item["total_time_ms"]), -float(item["max_duration_ms"]), str(item["op_name"])))

    pid_rows: List[Dict[str, Any]] = []
    for pid, bucket in sorted(pid_buckets.items(), key=lambda item: item[1]["total_nccl_time_ns"], reverse=True):
        best_op = None
        best_ns = -1
        for op_name, total_ns in (bucket.get("_ops") or {}).items():
            if int(total_ns) > best_ns:
                best_op = str(op_name)
                best_ns = int(total_ns)
        pid_rows.append(
            {
                "pid": int(pid),
                "total_nccl_time_ms": _ns_to_ms(int(bucket["total_nccl_time_ns"] or 0)),
                "nccl_event_count": int(bucket["nccl_event_count"] or 0),
                "max_duration_ms": _ns_to_ms(int(bucket["max_duration_ns"] or 0)),
                "top_nccl_op": best_op,
            }
        )

    return {
        "present": True,
        "source": str(selected_events[0]["source"]) if selected_events else None,
        "ops": ops[: int(limit)],
        "pids": pid_rows,
        "event_count": len(selected_events),
        "windows": [{"start_ns": int(s), "end_ns": int(e)} for s, e in windows],
        "notes": notes,
        "sql": sql,
    }


def correlate_nvlink_with_nccl(trace_db: TraceDB, nccl: Mapping[str, Any]) -> Dict[str, Any]:
    windows = [
        (int(item.get("start_ns") or 0), int(item.get("end_ns") or 0))
        for item in (nccl.get("windows") or [])
        if int(item.get("end_ns") or 0) > int(item.get("start_ns") or 0)
    ]
    if not windows:
        return {
            "present": False,
            "missing_counters": False,
            "rows": [],
            "timeseries": [],
            "notes": ["No NCCL windows available for NVLink correlation."],
            "capture_instructions": [],
            "sql": {},
        }

    metrics_table = _pick_table_with_required_cols(
        trace_db,
        candidates=("GPU_METRICS",),
        required_cols=("timestamp", "metricId", "value"),
        name_hint="GPU_METRICS",
    )
    target_table = _pick_table_with_required_cols(
        trace_db,
        candidates=("TARGET_INFO_GPU_METRICS",),
        required_cols=("metricId", "metricName"),
        name_hint="GPU_METRICS",
    )
    if not metrics_table or not target_table:
        return {
            "present": False,
            "missing_counters": True,
            "rows": [],
            "timeseries": [],
            "notes": ["GPU metric tables were not found in this export."],
            "capture_instructions": _metric_capture_instructions(include_nccl=bool(nccl.get("present"))),
            "sql": {},
        }

    tinfo = trace_db.schema.table(target_table)
    join_cond = "m.metricId = t.metricId"
    if tinfo.has("typeId"):
        join_cond += " AND m.typeId = t.typeId"
    predicates = ["LOWER(t.metricName) LIKE ?" for _ in _NVLINK_SIGNAL_PATTERNS]
    params = ["%{}%".format(pattern.lower()) for pattern in _NVLINK_SIGNAL_PATTERNS]
    type_select = "m.typeId AS metric_source_id," if trace_db.schema.table(metrics_table).has("typeId") else "NULL AS metric_source_id,"
    sql_metrics = (
        "SELECT {type_select} m.timestamp AS timestamp_ns, t.metricName AS metric_name, m.value AS metric_value "
        "FROM {metrics} m JOIN {target} t ON {join_cond} "
        "WHERE " + " OR ".join(predicates) + " "
        "ORDER BY metric_source_id, timestamp_ns"
    ).format(type_select=type_select, metrics=metrics_table, target=target_table, join_cond=join_cond)
    rows = trace_db.conn.execute(sql_metrics, tuple(params)).fetchall()
    if not rows:
        return {
            "present": False,
            "missing_counters": True,
            "rows": [],
            "timeseries": [],
            "notes": ["GPU metric tables exist, but no NVLink-related metrics were exported."],
            "capture_instructions": _metric_capture_instructions(include_nccl=bool(nccl.get("present"))),
            "sql": {"nvlink_metrics": sql_metrics},
        }

    sql: Dict[str, str] = {"nvlink_metrics": sql_metrics}
    grouped: Dict[Any, List[Tuple[int, float, str]]] = {}
    for row in rows:
        metric_source_id = row["metric_source_id"] if "metric_source_id" in row.keys() else None
        grouped.setdefault(metric_source_id, []).append(
            (int(row["timestamp_ns"] or 0), float(row["metric_value"] or 0.0), str(row["metric_name"] or ""))
        )

    report_rows: List[Dict[str, Any]] = []
    timeseries_rows: List[Dict[str, Any]] = []
    for metric_source_id, samples in grouped.items():
        samples.sort(key=lambda item: item[0])
        total_value = 0.0
        total_count = 0
        active_value = 0.0
        active_count = 0
        active_values: List[float] = []
        inactive_value = 0.0
        inactive_count = 0
        metric_names = sorted({sample[2] for sample in samples})
        xs: List[float] = []
        ys: List[float] = []
        for timestamp_ns, metric_value, _metric_name in samples:
            active = 1.0 if _interval_overlap_ns(windows, int(timestamp_ns), int(timestamp_ns) + 1) > 0 else 0.0
            total_value += float(metric_value)
            total_count += 1
            xs.append(active)
            ys.append(float(metric_value))
            timeseries_rows.append(
                {
                    "metric_source_id": metric_source_id,
                    "metric_name": str(_metric_name),
                    "timestamp_ns": int(timestamp_ns),
                    "timestamp_ms": _ns_to_ms(int(timestamp_ns)),
                    "metric_value": float(metric_value),
                    "nccl_active": bool(active > 0.0),
                }
            )
            if active > 0.0:
                active_value += float(metric_value)
                active_count += 1
                active_values.append(float(metric_value))
            else:
                inactive_value += float(metric_value)
                inactive_count += 1

        mean_x = _safe_div(sum(xs), float(len(xs))) if xs else 0.0
        mean_y = _safe_div(sum(ys), float(len(ys))) if ys else 0.0
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        corr = _safe_div(cov, math.sqrt(var_x * var_y)) if var_x > 0 and var_y > 0 else 0.0
        report_rows.append(
            {
                "metric_source_id": metric_source_id,
                "metric_names": ", ".join(metric_names),
                "samples": int(total_count),
                "samples_during_nccl": int(active_count),
                "avg_metric_during_nccl": _safe_div(active_value, float(active_count)),
                "avg_metric_outside_nccl": _safe_div(inactive_value, float(inactive_count)),
                "max_metric_during_nccl": max(active_values) if active_values else 0.0,
                "nccl_activity_correlation": float(corr),
            }
        )
    report_rows.sort(
        key=lambda item: (
            -float(item.get("avg_metric_during_nccl") or 0.0),
            -float(item.get("nccl_activity_correlation") or 0.0),
            str(item.get("metric_source_id") or ""),
        )
    )
    return {
        "present": True,
        "missing_counters": False,
        "rows": report_rows,
        "timeseries": timeseries_rows,
        "notes": [
            "Correlation is computed between sampled NVLink metric values and a binary NCCL-active timeline derived from NCCL windows.",
            "GPU Metrics are device-level samples; they are not process-attributed in the SQLite export.",
        ],
        "capture_instructions": [],
        "sql": sql,
    }


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def estimate_gpu_idle_gaps(
    trace_db: TraceDB,
    *,
    top_n_gaps: int = 50,
    per_device: bool = True,
) -> Dict[str, Any]:
    ktable = trace_db.schema.kernel_table
    if not ktable:
        return {"table": None, "devices": [], "gaps": [], "notes": ["No kernel table found."], "sql": {}}

    kinfo = trace_db.schema.table(ktable)
    has_device = kinfo.has("deviceId")
    device_expr = "deviceId" if (per_device and has_device) else "-1"
    sql = "SELECT start, end, {dev} AS device_id FROM {t} ORDER BY device_id, start".format(dev=device_expr, t=ktable)
    rows = trace_db.conn.execute(sql).fetchall()

    by_dev: Dict[int, List[Tuple[int, int]]] = {}
    for r in rows:
        s = int(r["start"] or 0)
        e = int(r["end"] or 0)
        dev = int(r["device_id"]) if r["device_id"] is not None else -1
        by_dev.setdefault(dev, []).append((s, e))

    devices: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []

    for dev, ints in sorted(by_dev.items(), key=lambda kv: kv[0]):
        if not ints:
            continue
        window_start = min(s for s, _ in ints)
        window_end = max(e for _, e in ints)
        window_ns = max(0, window_end - window_start)
        merged = _merge_intervals([(s, e) for s, e in ints if e >= s])
        busy_ns = sum(max(0, e - s) for s, e in merged)
        idle_ns = max(0, window_ns - busy_ns)
        devices.append(
            {
                "device_id": dev,
                "window_ms": _ns_to_ms(window_ns),
                "busy_ms": _ns_to_ms(busy_ns),
                "idle_ms": _ns_to_ms(idle_ns),
                "idle_pct_of_window": (_safe_div(float(idle_ns), float(window_ns)) * 100.0) if window_ns else 0.0,
            }
        )

        for (s1, e1), (s2, _e2) in zip(merged, merged[1:]):
            if s2 > e1:
                gap_ns = s2 - e1
                gaps.append({"device_id": dev, "gap_start_ns": e1, "gap_end_ns": s2, "gap_ms": _ns_to_ms(gap_ns)})

    gaps.sort(key=lambda g: float(g["gap_ms"]), reverse=True)
    gaps = gaps[: int(top_n_gaps)]
    return {"table": ktable, "devices": devices, "gaps": gaps, "notes": [], "sql": {"events": sql}}


def timeline_events(
    trace_db: TraceDB,
    *,
    limit: int = 50,
    include_nccl: bool = True,
) -> Dict[str, Any]:
    """Top timeline events across GPU kernels, CPU stalls, NCCL windows, and GPU idle gaps."""

    events: List[Dict[str, Any]] = []
    notes: List[str] = []
    sql: Dict[str, str] = {}
    stable = trace_db.schema.string_table

    total_gpu_time_ns = 0
    total_cpu_time_ns = 0

    ktable = trace_db.schema.kernel_table
    if ktable:
        total_gpu_time_ns = int(_fetch_one(trace_db.conn, "SELECT SUM(end-start) FROM {t}".format(t=ktable)) or 0)
        kinfo = trace_db.schema.table(ktable)
        name_col = "demangledName" if kinfo.has("demangledName") else ("shortName" if kinfo.has("shortName") else None)
        if name_col:
            pid_expr, _pid_source, _ = _pid_expr_for_table("k", kinfo)
            pid_select = "{pid} AS pid,".format(pid=pid_expr) if pid_expr else "NULL AS pid,"
            stream_select = "k.streamId AS stream_id," if kinfo.has("streamId") else "NULL AS stream_id,"
            device_select = "k.deviceId AS device_id," if kinfo.has("deviceId") else "NULL AS device_id,"
            name_expr = "k.{c}".format(c=name_col)
            join = ""
            if stable and not trace_db.schema.is_text_column(ktable, name_col):
                join = " LEFT JOIN {s} sk ON sk.id = k.{c} ".format(s=stable, c=name_col)
                name_expr = "COALESCE(sk.value, CAST(k.{c} AS TEXT))".format(c=name_col)
            sql_kernels = (
                "SELECT {pid_select} {stream_select} {device_select} "
                "{name} AS event_name, k.start AS start_ns, k.end AS end_ns, "
                "(k.end-k.start) AS duration_ns "
                "FROM {table} k {join} "
                "WHERE k.end IS NOT NULL AND k.end > k.start "
                "ORDER BY duration_ns DESC "
                "LIMIT ?"
            ).format(
                pid_select=pid_select,
                stream_select=stream_select,
                device_select=device_select,
                name=name_expr,
                table=ktable,
                join=join,
            )
            sql["timeline_gpu_kernels"] = sql_kernels
            for row in trace_db.conn.execute(sql_kernels, (int(max(100, limit * 6)),)).fetchall():
                event_name = str(row["event_name"] or "kernel")
                is_nccl = _looks_like_nccl(event_name)
                if is_nccl and not include_nccl:
                    continue
                stream_id = int(row["stream_id"]) if row["stream_id"] is not None else None
                device_id = int(row["device_id"]) if row["device_id"] is not None else None
                pid = int(row["pid"]) if row["pid"] is not None else None
                lane = (
                    "GPU stream {}".format(stream_id)
                    if stream_id is not None
                    else ("GPU device {}".format(device_id) if device_id is not None else "GPU")
                )
                event_class = "nccl" if is_nccl else "cuda_kernel"
                if (not is_nccl) and _name_matches_patterns(event_name, _MEMORY_KERNEL_PATTERNS):
                    event_class = "memory_kernel"
                start_ns = int(row["start_ns"] or 0)
                end_ns = int(row["end_ns"] or 0)
                duration_ns = max(0, int(row["duration_ns"] or 0))
                events.append(
                    {
                        "event_name": event_name,
                        "event_class": event_class,
                        "lane": lane,
                        "pid": pid,
                        "stream_id": stream_id,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "duration_ns": duration_ns,
                        "duration_ms": _ns_to_ms(duration_ns),
                    }
                )

    rtable = trace_db.schema.runtime_table
    if rtable:
        total_cpu_time_ns = int(_fetch_one(trace_db.conn, "SELECT SUM(end-start) FROM {t}".format(t=rtable)) or 0)
        rinfo = trace_db.schema.table(rtable)
        name_col = "nameId" if rinfo.has("nameId") else ("name" if rinfo.has("name") else None)
        if name_col:
            pid_expr, _pid_source, _ = _pid_expr_for_table("r", rinfo)
            pid_select = "{pid} AS pid,".format(pid=pid_expr) if pid_expr else "NULL AS pid,"
            tid_select = "((CAST(r.globalTid AS INT)) % 16777216) AS tid," if rinfo.has("globalTid") else "NULL AS tid,"
            name_expr = "r.{c}".format(c=name_col)
            join = ""
            if stable and not trace_db.schema.is_text_column(rtable, name_col):
                join = " LEFT JOIN {s} sr ON sr.id = r.{c} ".format(s=stable, c=name_col)
                name_expr = "COALESCE(sr.value, CAST(r.{c} AS TEXT))".format(c=name_col)

            stall_patterns = sorted(set(_SYNC_BARRIER_PATTERNS + _BLOCKING_COPY_PATTERNS + _HOST_WAIT_PATTERNS + ("wait", "synchronize")))
            where_parts = ["LOWER({name}) LIKE ?".format(name=name_expr) for _ in stall_patterns]
            params: List[Any] = ["%{}%".format(pattern.lower()) for pattern in stall_patterns]
            sql_runtime = (
                "SELECT {pid_select} {tid_select} {name} AS event_name, "
                "r.start AS start_ns, r.end AS end_ns, (r.end-r.start) AS duration_ns "
                "FROM {table} r {join} "
                "WHERE r.end IS NOT NULL AND r.end > r.start AND (" + " OR ".join(where_parts) + ") "
                "ORDER BY duration_ns DESC "
                "LIMIT ?"
            ).format(
                pid_select=pid_select,
                tid_select=tid_select,
                name=name_expr,
                table=rtable,
                join=join,
            )
            sql["timeline_cpu_stalls"] = sql_runtime
            for row in trace_db.conn.execute(sql_runtime, tuple(params) + (int(max(100, limit * 4)),)).fetchall():
                pid = int(row["pid"]) if row["pid"] is not None else None
                tid = int(row["tid"]) if row["tid"] is not None else None
                lane = (
                    "CPU pid {} tid {}".format(pid, tid)
                    if (pid is not None and tid is not None)
                    else ("CPU pid {}".format(pid) if pid is not None else "CPU")
                )
                start_ns = int(row["start_ns"] or 0)
                end_ns = int(row["end_ns"] or 0)
                duration_ns = max(0, int(row["duration_ns"] or 0))
                events.append(
                    {
                        "event_name": str(row["event_name"] or "cpu_stall"),
                        "event_class": "cpu_stall",
                        "lane": lane,
                        "pid": pid,
                        "stream_id": None,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "duration_ns": duration_ns,
                        "duration_ms": _ns_to_ms(duration_ns),
                    }
                )

    if include_nccl:
        nccl = detect_nccl_ops(trace_db, limit=max(20, limit))
        for idx, window in enumerate(nccl.get("windows") or []):
            start_ns = int(window.get("start_ns") or 0)
            end_ns = int(window.get("end_ns") or 0)
            if end_ns <= start_ns:
                continue
            duration_ns = end_ns - start_ns
            events.append(
                {
                    "event_name": "NCCL window {}".format(idx + 1),
                    "event_class": "nccl",
                    "lane": "GPU NCCL",
                    "pid": None,
                    "stream_id": None,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ns": duration_ns,
                    "duration_ms": _ns_to_ms(duration_ns),
                }
            )

    idle = estimate_gpu_idle_gaps(trace_db, top_n_gaps=max(20, limit))
    for gap in idle.get("gaps") or []:
        start_ns = int(gap.get("gap_start_ns") or 0)
        end_ns = int(gap.get("gap_end_ns") or 0)
        if end_ns <= start_ns:
            continue
        duration_ns = end_ns - start_ns
        device_id = gap.get("device_id")
        events.append(
            {
                "event_name": "GPU idle gap",
                "event_class": "idle",
                "lane": "GPU device {}".format(device_id) if device_id is not None else "GPU",
                "pid": None,
                "stream_id": None,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": duration_ns,
                "duration_ms": _ns_to_ms(duration_ns),
            }
        )

    events.sort(
        key=lambda item: (
            -int(item.get("duration_ns") or 0),
            int(item.get("start_ns") or 0),
            str(item.get("event_name") or ""),
        )
    )
    top_events = events[: int(limit)]
    t0_ns = min((int(item.get("start_ns") or 0) for item in top_events), default=0)
    for item in top_events:
        item["start_ms"] = _ns_to_ms(int(item.get("start_ns") or 0) - int(t0_ns))
        item["end_ms"] = _ns_to_ms(int(item.get("end_ns") or 0) - int(t0_ns))

    if not top_events:
        notes.append("No timeline events available from this trace export.")

    return {
        "present": bool(top_events),
        "events": top_events,
        "t0_ns": int(t0_ns),
        "total_gpu_time_ms": _ns_to_ms(total_gpu_time_ns),
        "total_cpu_time_ms": _ns_to_ms(total_cpu_time_ns),
        "notes": notes,
        "sql": sql,
    }


def nvtx_breakdown(trace_db: TraceDB, *, limit: int = 50) -> Dict[str, Any]:
    ntable = trace_db.schema.nvtx_table
    stable = trace_db.schema.string_table
    if not ntable:
        return {"table": None, "ranges": [], "instances": [], "notes": ["No NVTX table found."], "sql": {}}

    ninfo = trace_db.schema.table(ntable)
    if not ninfo.has("end"):
        return {"table": ntable, "ranges": [], "instances": [], "notes": ["NVTX table missing end column."], "sql": {}}

    has_text = ninfo.has("text")
    has_text_id = ninfo.has("textId")
    has_tid = ninfo.has("globalTid")

    range_expr = "e.text" if has_text else ("e.textId" if has_text_id else "''")
    join = ""
    if has_text_id and stable and not trace_db.schema.is_text_column(ntable, "textId"):
        join = " LEFT JOIN {s} s ON s.id = e.textId ".format(s=stable)
        range_expr = "COALESCE(e.text, s.value)" if has_text else "s.value"

    sql_ranges = (
        "SELECT {name} AS range_name, COUNT(*) AS count, "
        "SUM(e.end - e.start) AS total_ns, AVG(e.end-e.start) AS avg_ns "
        "FROM {t} e {join} "
        "WHERE e.end IS NOT NULL AND e.end > e.start "
        "GROUP BY range_name "
        "ORDER BY total_ns DESC "
        "LIMIT ?"
    ).format(name=range_expr, t=ntable, join=join)
    rows = trace_db.conn.execute(sql_ranges, (int(limit),)).fetchall()

    ranges: List[Dict[str, Any]] = []
    for r in rows:
        total_ns = int(r["total_ns"] or 0)
        ranges.append(
            {
                "range_name": str(r["range_name"]),
                "count": int(r["count"] or 0),
                "total_time_ms": _ns_to_ms(total_ns),
                "avg_duration_us": _ns_to_us(float(r["avg_ns"] or 0.0)),
            }
        )

    instances: List[Dict[str, Any]] = []
    sql_instances = None
    if has_tid:
        sql_instances = (
            "SELECT e.start AS start_ns, e.end AS end_ns, {name} AS range_name, e.globalTid AS global_tid "
            "FROM {t} e {join} "
            "WHERE e.end IS NOT NULL AND e.end > e.start "
            "ORDER BY (e.end - e.start) DESC "
            "LIMIT ?"
        ).format(name=range_expr, t=ntable, join=join)
        for r in trace_db.conn.execute(sql_instances, (int(limit),)).fetchall():
            pid, tid = decode_global_tid(int(r["global_tid"]) if r["global_tid"] is not None else None)
            instances.append(
                {
                    "range_name": str(r["range_name"]),
                    "start_ns": int(r["start_ns"] or 0),
                    "end_ns": int(r["end_ns"] or 0),
                    "dur_ms": _ns_to_ms(int(r["end_ns"] or 0) - int(r["start_ns"] or 0)),
                    "pid": pid,
                    "tid": tid,
                }
            )

    notes: List[str] = []
    if has_text_id and not stable:
        notes.append("NVTX events may reference textId, but StringIds table was not found (range_name may be numeric IDs).")
    if not instances and not has_tid:
        notes.append("NVTX per-instance export omitted (globalTid missing).")

    return {"table": ntable, "ranges": ranges, "instances": instances, "notes": notes, "sql": {"ranges": sql_ranges, "instances": sql_instances}}


def _pid_expr_for_table_cols(alias: str, cols: Sequence[str]) -> Optional[Tuple[str, str]]:
    """
    Return (pid_expr_sql, pid_source_column) for a table alias, if possible.

    Nsight Systems commonly stores:
    - kernel globalPid: serialized pid in upper bits (pid = (globalPid >> 24) & 0xFFFFFF)
    - runtime/nvtx globalTid: serialized (pid,tid) (pid = (globalTid >> 24) & 0xFFFFFF)
    """
    if "pid" in cols:
        return ("{}.pid".format(alias), "pid")
    if "processId" in cols:
        return ("{}.processId".format(alias), "processId")
    if "globalPid" in cols:
        return ("(({}.globalPid >> 24) & 16777215)".format(alias), "globalPid")
    if "globalTid" in cols:
        return ("(({}.globalTid >> 24) & 16777215)".format(alias), "globalTid")
    return None


def per_pid_breakdown(
    trace_db: TraceDB,
    *,
    top_pids: int = 5,
    kernel_limit: int = 15,
    tiny_kernel_us: float = 5.0,
    tiny_kernel_limit: int = 10,
) -> Dict[str, Any]:
    """
    Multi-process attribution (best-effort).

    Produces per-PID: top kernels, launch-storm stats, sync-like runtime calls, NVTX range totals.
    Falls back to global-only if PID columns are missing.
    """
    ktable = trace_db.schema.kernel_table
    if not ktable:
        return {"present": False, "reason": "no_kernel_table", "top_pids": [], "pids": [], "notes": []}

    kinfo = trace_db.schema.table(ktable)
    pid_info = _pid_expr_for_table_cols("k", list(kinfo.columns))
    if not pid_info:
        return {"present": False, "reason": "no_pid_column", "top_pids": [], "pids": [], "notes": ["Kernel table has no PID/globalPid/processId column."]}

    pid_expr, pid_source = pid_info
    stable = trace_db.schema.string_table
    name_col = "demangledName" if kinfo.has("demangledName") else ("shortName" if kinfo.has("shortName") else None)

    # Total kernel time (global) for % attribution.
    total_kernel_time_ns = int(_fetch_one(trace_db.conn, "SELECT SUM(end-start) FROM {t}".format(t=ktable)) or 0)

    sql_top_pids = (
        "SELECT {pid} AS pid, SUM(k.end-k.start) AS total_ns, COUNT(*) AS launches "
        "FROM {t} k "
        "WHERE k.end > k.start AND {pid} IS NOT NULL "
        "GROUP BY pid "
        "ORDER BY total_ns DESC "
        "LIMIT ?"
    ).format(pid=pid_expr, t=ktable)
    top_rows = trace_db.conn.execute(sql_top_pids, (int(top_pids),)).fetchall()
    top: List[Dict[str, Any]] = []
    pids: List[int] = []
    for r in top_rows:
        pid = int(r["pid"])
        pids.append(pid)
        total_ns = int(r["total_ns"] or 0)
        top.append(
            {
                "pid": pid,
                "total_kernel_time_ms": _ns_to_ms(total_ns),
                "total_kernel_time_ns": total_ns,
                "kernel_launches": int(r["launches"] or 0),
                "pct_of_total_kernel_time": (_safe_div(float(total_ns), float(total_kernel_time_ns)) * 100.0)
                if total_kernel_time_ns
                else 0.0,
            }
        )

    # Helpers for per-pid kernels and tiny-kernel suspects.
    def _top_kernels_for_pid(pid: int) -> Dict[str, Any]:
        if not name_col:
            return {"table": ktable, "kernels": [], "notes": ["Kernel table missing demangledName/shortName."]}
        join = ""
        group_name_expr = "k.{c}".format(c=name_col)
        if stable and not trace_db.schema.is_text_column(ktable, name_col):
            join = " JOIN {s} s ON s.id = k.{c} ".format(s=stable, c=name_col)
            group_name_expr = "s.value"

        has_device = kinfo.has("deviceId")
        device_expr = "k.deviceId" if has_device else "-1"

        sql_agg = (
            "SELECT {name} AS kernel_name, {dev} AS device_id, "
            "COUNT(*) AS call_count, "
            "SUM(k.end-k.start) AS total_time_ns, "
            "AVG(k.end-k.start) AS avg_time_ns "
            "FROM {t} k {join} "
            "WHERE {pid_expr} = ? AND k.end > k.start "
            "GROUP BY kernel_name, device_id "
            "ORDER BY total_time_ns DESC "
            "LIMIT ?"
        ).format(name=group_name_expr, dev=device_expr, t=ktable, join=join, pid_expr=pid_expr)
        rows = trace_db.conn.execute(sql_agg, (int(pid), int(kernel_limit))).fetchall()
        kernels: List[Dict[str, Any]] = []
        for rr in rows:
            total_ns = int(rr["total_time_ns"] or 0)
            kernels.append(
                {
                    "kernel_name": str(rr["kernel_name"]),
                    "device_id": int(rr["device_id"]) if rr["device_id"] is not None else None,
                    "call_count": int(rr["call_count"] or 0),
                    "total_time_ms": _ns_to_ms(total_ns),
                    "avg_duration_us": _ns_to_us(float(rr["avg_time_ns"] or 0.0)),
                }
            )

        tiny: List[Dict[str, Any]] = []
        if name_col:
            join2 = ""
            name_expr = "k.{c}".format(c=name_col)
            if stable and not trace_db.schema.is_text_column(ktable, name_col):
                join2 = " JOIN {s} s ON s.id = k.{c} ".format(s=stable, c=name_col)
                name_expr = "s.value"
            sql_tiny = (
                "SELECT {name} AS kernel_name, COUNT(*) AS call_count, AVG(k.end-k.start) AS avg_dur_ns "
                "FROM {t} k {join} "
                "WHERE {pid_expr} = ? AND (k.end-k.start) <= ? "
                "GROUP BY kernel_name "
                "ORDER BY call_count DESC "
                "LIMIT ?"
            ).format(name=name_expr, t=ktable, join=join2, pid_expr=pid_expr)
            for tr in trace_db.conn.execute(sql_tiny, (int(pid), int(tiny_kernel_us * 1000.0), int(tiny_kernel_limit))).fetchall():
                tiny.append(
                    {
                        "kernel_name": str(tr["kernel_name"]),
                        "call_count": int(tr["call_count"] or 0),
                        "avg_duration_us": _ns_to_us(float(tr["avg_dur_ns"] or 0.0)),
                    }
                )

        return {"table": ktable, "kernels": kernels, "tiny_kernels": tiny}

    def _launch_storm_for_pid(pid: int) -> Dict[str, Any]:
        # Avoid fetching all durations: compute window + percentiles via count + ordered OFFSET.
        sql_cnt = "SELECT COUNT(*) FROM {t} k WHERE {pid_expr} = ? AND k.end > k.start".format(t=ktable, pid_expr=pid_expr)
        n = int(_fetch_one(trace_db.conn, sql_cnt, (int(pid),)) or 0)
        if n <= 0:
            return {
                "total_launches": 0,
                "window_s": 0.0,
                "launches_per_s": 0.0,
                "median_kernel_us": None,
                "p50_kernel_us": None,
                "p90_kernel_us": None,
                "p99_kernel_us": None,
                "pct_under_5us": None,
                "pct_under_10us": None,
                "pct_under_20us": None,
                "is_launch_storm": None,
            }
        sql_win = "SELECT MIN(k.start) AS s, MAX(k.end) AS e FROM {t} k WHERE {pid_expr} = ?".format(t=ktable, pid_expr=pid_expr)
        row = trace_db.conn.execute(sql_win, (int(pid),)).fetchone()
        s = int(row["s"] or 0) if row else 0
        e = int(row["e"] or 0) if row else 0
        window_ns = max(1, e - s)
        window_s = float(window_ns) / 1_000_000_000.0

        def _dur_at_offset(offset: int) -> Optional[float]:
            sql_q = (
                "SELECT (k.end-k.start) AS dur_ns FROM {t} k "
                "WHERE {pid_expr} = ? AND k.end > k.start "
                "ORDER BY dur_ns "
                "LIMIT 1 OFFSET ?"
            ).format(t=ktable, pid_expr=pid_expr)
            v = _fetch_one(trace_db.conn, sql_q, (int(pid), int(offset)))
            return float(v) if v is not None else None

        def _offset(q: float) -> int:
            # nearest-rank style on [0, n-1]
            if n <= 1:
                return 0
            return int(round(float(q) * float(n - 1)))

        p50_ns = _dur_at_offset(_offset(0.50))
        p90_ns = _dur_at_offset(_offset(0.90))
        p99_ns = _dur_at_offset(_offset(0.99))

        # Count-under thresholds (%).
        def _pct_under(us: float) -> float:
            sql_u = (
                "SELECT COUNT(*) FROM {t} k WHERE {pid_expr} = ? AND k.end > k.start AND (k.end-k.start) < ?"
            ).format(t=ktable, pid_expr=pid_expr)
            c = int(_fetch_one(trace_db.conn, sql_u, (int(pid), int(us * 1000.0))) or 0)
            return (_safe_div(float(c), float(n)) * 100.0) if n else 0.0

        pct_5 = _pct_under(5.0)
        pct_10 = _pct_under(10.0)
        pct_20 = _pct_under(20.0)

        # Classification thresholds live in heuristics.py.
        try:
            from .heuristics import LAUNCH_STORM_THRESHOLDS, classify_launch_storm
        except Exception:
            LAUNCH_STORM_THRESHOLDS = {}
            classify_launch_storm = None

        launches_per_s = _safe_div(float(n), float(window_s))
        p50_us = _ns_to_us(p50_ns or 0.0) if p50_ns is not None else None
        is_storm = bool(classify_launch_storm(float(launches_per_s), float(p50_us or 0.0))) if classify_launch_storm else None

        # Keep the historical field name for compatibility.
        med_us = p50_us

        return {
            "total_launches": int(n),
            "window_s": float(window_s),
            "launches_per_s": float(launches_per_s),
            "median_kernel_us": float(med_us) if med_us is not None else None,
            "p50_kernel_us": float(p50_us) if p50_us is not None else None,
            "p90_kernel_us": float(_ns_to_us(p90_ns or 0.0)) if p90_ns is not None else None,
            "p99_kernel_us": float(_ns_to_us(p99_ns or 0.0)) if p99_ns is not None else None,
            "pct_under_5us": float(pct_5),
            "pct_under_10us": float(pct_10),
            "pct_under_20us": float(pct_20),
            "is_launch_storm": is_storm,
            "storm_thresholds": LAUNCH_STORM_THRESHOLDS,
        }

    # Sync-like calls per PID (runtime/globalTid).
    def _sync_for_pid(pid: int) -> Dict[str, Any]:
        rtable = trace_db.schema.runtime_table
        if not rtable:
            return {"present": False, "table": None, "sync_calls": [], "notes": ["No runtime table."]}
        rinfo = trace_db.schema.table(rtable)
        pid_rt = _pid_expr_for_table_cols("r", list(rinfo.columns))
        if not pid_rt:
            return {"present": False, "table": rtable, "sync_calls": [], "notes": ["Runtime table missing globalTid/pid/processId."]}
        pid_expr_rt, _pid_src_rt = pid_rt

        name_col_rt = "nameId" if rinfo.has("nameId") else ("name" if rinfo.has("name") else None)
        if not name_col_rt:
            return {"present": False, "table": rtable, "sync_calls": [], "notes": ["Runtime table missing name/nameId."]}

        join = ""
        name_expr = "r.{c}".format(c=name_col_rt)
        if stable and not trace_db.schema.is_text_column(rtable, name_col_rt):
            join = " JOIN {s} s ON s.id = r.{c} ".format(s=stable, c=name_col_rt)
            name_expr = "s.value"

        sync_keywords = [
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
            "cudaStreamWaitEvent",
            "cudaEventQuery",
            "cuCtxSynchronize",
            "cuStreamSynchronize",
            "cuEventSynchronize",
            "cuStreamWaitEvent",
        ]
        where_parts = ["({expr} LIKE ?)".format(expr=name_expr) for _ in range(len(sync_keywords) + 2)]
        params: List[Any] = ["%{}%".format(k) for k in sync_keywords] + ["%Wait%", "%Synchronize%"]

        sql = (
            "SELECT {name} AS api_name, COUNT(*) AS call_count, "
            "SUM(r.end-r.start) AS total_time_ns, AVG(r.end-r.start) AS avg_time_ns "
            "FROM {t} r {join} "
            "WHERE {pid_expr} = ? AND (" + " OR ".join(where_parts) + ") "
            "GROUP BY api_name "
            "ORDER BY total_time_ns DESC "
            "LIMIT 50"
        ).format(name=name_expr, t=rtable, join=join, pid_expr=pid_expr_rt)
        rows = trace_db.conn.execute(sql, (int(pid), *params)).fetchall()
        out: List[Dict[str, Any]] = []
        for rr in rows:
            total_ns = int(rr["total_time_ns"] or 0)
            out.append(
                {
                    "api_name": str(rr["api_name"]),
                    "call_count": int(rr["call_count"] or 0),
                    "total_time_ms": _ns_to_ms(total_ns),
                    "avg_duration_us": _ns_to_us(float(rr["avg_time_ns"] or 0.0)),
                }
            )
        return {"present": True, "table": rtable, "sync_calls": out, "notes": [], "sql": sql}

    # NVTX per PID (globalTid).
    def _nvtx_for_pid(pid: int) -> Dict[str, Any]:
        ntable = trace_db.schema.nvtx_table
        if not ntable:
            return {"present": False, "table": None, "ranges": [], "notes": ["No NVTX table."]}
        ninfo = trace_db.schema.table(ntable)
        if not (ninfo.has("start") and ninfo.has("end")):
            return {"present": False, "table": ntable, "ranges": [], "notes": ["NVTX table missing start/end."]}
        pid_nv = _pid_expr_for_table_cols("e", list(ninfo.columns))
        if not pid_nv:
            return {"present": False, "table": ntable, "ranges": [], "notes": ["NVTX table missing globalTid/pid/processId."]}
        pid_expr_nv, _pid_src_nv = pid_nv

        has_text = ninfo.has("text")
        has_text_id = ninfo.has("textId")
        range_expr = "e.text" if has_text else ("e.textId" if has_text_id else "''")
        join = ""
        if has_text_id and stable and not trace_db.schema.is_text_column(ntable, "textId"):
            join = " LEFT JOIN {s} s ON s.id = e.textId ".format(s=stable)
            range_expr = "COALESCE(e.text, s.value)" if has_text else "s.value"

        sql_ranges = (
            "SELECT {name} AS range_name, COUNT(*) AS count, "
            "SUM(e.end-e.start) AS total_ns, AVG(e.end-e.start) AS avg_ns "
            "FROM {t} e {join} "
            "WHERE {pid_expr} = ? AND e.end IS NOT NULL AND e.end > e.start "
            "GROUP BY range_name "
            "ORDER BY total_ns DESC "
            "LIMIT 50"
        ).format(name=range_expr, t=ntable, join=join, pid_expr=pid_expr_nv)
        rows = trace_db.conn.execute(sql_ranges, (int(pid),)).fetchall()
        ranges: List[Dict[str, Any]] = []
        for rr in rows:
            total_ns = int(rr["total_ns"] or 0)
            ranges.append(
                {
                    "range_name": str(rr["range_name"]),
                    "count": int(rr["count"] or 0),
                    "total_time_ms": _ns_to_ms(total_ns),
                    "avg_duration_us": _ns_to_us(float(rr["avg_ns"] or 0.0)),
                }
            )
        return {"present": True, "table": ntable, "ranges": ranges, "notes": [], "sql": sql_ranges}

    per_pid: List[Dict[str, Any]] = []
    for pid in pids:
        per_pid.append(
            {
                "pid": pid,
                "top_kernels": _top_kernels_for_pid(pid),
                "launch_storm": _launch_storm_for_pid(pid),
                "sync": _sync_for_pid(pid),
                "nvtx": _nvtx_for_pid(pid),
            }
        )

    return {
        "present": True,
        "pid_source": pid_source,
        "top_pids": top,
        "pids": per_pid,
        "notes": [],
        "sql": {"top_pids": sql_top_pids},
    }


def nvtx_kernel_time_by_range(trace_db: TraceDB, *, limit: int = 50) -> Dict[str, Any]:
    """
    Best-effort attribution of *GPU kernel time* to NVTX ranges using:

    kernel.correlationId -> runtime.correlationId (launch site) -> runtime.globalTid -> enclosing NVTX range on same globalTid.

    This is intentionally conservative:
    - Only kernels with a non-null correlationId that maps to a runtime row and an enclosing NVTX range are attributed.
    - Many traces will have partial coverage depending on what Nsight Systems exported.
    """

    ktable = trace_db.schema.kernel_table
    rtable = trace_db.schema.runtime_table
    ntable = trace_db.schema.nvtx_table
    stable = trace_db.schema.string_table

    if not ktable or not rtable or not ntable:
        return {
            "present": False,
            "ranges": [],
            "notes": ["Need kernel + runtime + NVTX tables for NVTX→kernel attribution."],
            "sql": {},
        }

    kinfo = trace_db.schema.table(ktable)
    rinfo = trace_db.schema.table(rtable)
    ninfo = trace_db.schema.table(ntable)

    def pick_col(info: Any, candidates: Sequence[str]) -> Optional[str]:
        for c in candidates:
            if info.has(c):
                return c
        return None

    k_cid = pick_col(kinfo, ["correlationId", "correlationID", "correlation_id"])
    r_cid = pick_col(rinfo, ["correlationId", "correlationID", "correlation_id"])
    r_gt = pick_col(rinfo, ["globalTid", "globalTID", "global_tid"])
    n_gt = pick_col(ninfo, ["globalTid", "globalTID", "global_tid"])

    missing: List[str] = []
    if k_cid is None:
        missing.append("{}/correlationId".format(ktable))
    if r_cid is None:
        missing.append("{}/correlationId".format(rtable))
    if r_gt is None:
        missing.append("{}/globalTid".format(rtable))
    if n_gt is None:
        missing.append("{}/globalTid".format(ntable))
    if not ninfo.has("end"):
        missing.append("{}/end".format(ntable))
    if missing:
        return {
            "present": False,
            "ranges": [],
            "notes": ["Missing required columns for NVTX→kernel mapping: {}".format(", ".join(missing))],
            "sql": {},
        }

    has_text = ninfo.has("text")
    has_text_id = ninfo.has("textId")
    range_expr = "n.text" if has_text else ("n.textId" if has_text_id else "''")
    join = ""
    if has_text_id and stable and not trace_db.schema.is_text_column(ntable, "textId"):
        join = " LEFT JOIN {s} s ON s.id = n.textId ".format(s=stable)
        range_expr = "COALESCE(n.text, s.value)" if has_text else "s.value"

    nvtx_filter = "WHERE n.end IS NOT NULL AND n.end > n.start"
    if ninfo.has("eventType"):
        # Nsight Systems typically uses 59 (push/pop range) and 60 (start/end range)
        nvtx_filter += " AND n.eventType IN (59, 60)"

    # Total kernel time for context.
    total_kernel_time_ns = int(_fetch_one(trace_db.conn, "SELECT SUM(k.end - k.start) FROM {t} k".format(t=ktable)) or 0)

    sql = """
        WITH runtime AS (
            SELECT
            r.{r_cid} AS correlation_id,
                r.start AS r_start,
                r.end AS r_end,
            r.{r_gt} AS global_tid
        FROM {rtable} r
        WHERE r.{r_cid} IS NOT NULL AND r.{r_gt} IS NOT NULL AND r.end IS NOT NULL
        ),
        nvtx AS (
            SELECT
                n.start AS n_start,
                n.end AS n_end,
            n.{n_gt} AS global_tid,
            {range_expr} AS range_name
        FROM {ntable} n
        {join}
        {nvtx_filter}
          AND n.{n_gt} IS NOT NULL
    ),
    mapped AS (
            SELECT
            (k.end - k.start) AS dur_ns,
                (
                SELECT n2.range_name
                    FROM nvtx n2
                    WHERE n2.global_tid = runtime.global_tid
                      AND n2.n_start <= runtime.r_start
                      AND n2.n_end >= runtime.r_end
                    ORDER BY n2.n_start DESC
                    LIMIT 1
            ) AS range_name
        FROM {ktable} k
        JOIN runtime ON runtime.correlation_id = k.{k_cid}
        WHERE k.{k_cid} IS NOT NULL
          AND k.end IS NOT NULL AND k.end > k.start
    )
    SELECT
        range_name,
        COUNT(*) AS kernel_count,
        SUM(dur_ns) AS total_dur_ns,
        AVG(dur_ns) AS avg_dur_ns
    FROM mapped
    WHERE range_name IS NOT NULL
    GROUP BY range_name
    ORDER BY total_dur_ns DESC
    LIMIT ?;
    """.format(
        rtable=rtable,
        r_cid=r_cid,
        r_gt=r_gt,
        ntable=ntable,
        n_gt=n_gt,
        range_expr=range_expr,
        join=join,
        nvtx_filter=nvtx_filter,
        ktable=ktable,
        k_cid=k_cid,
    )

    rows = trace_db.conn.execute(sql, (int(limit),)).fetchall()
    ranges: List[Dict[str, Any]] = []
    mapped_kernel_time_ns = 0
    mapped_kernel_count = 0
    for r in rows:
        total_ns = int(r["total_dur_ns"] or 0)
        mapped_kernel_time_ns += total_ns
        mapped_kernel_count += int(r["kernel_count"] or 0)
        ranges.append(
            {
                "range_name": str(r["range_name"]),
                "kernel_count": int(r["kernel_count"] or 0),
                "total_kernel_time_ns": total_ns,
                "total_kernel_time_ms": _ns_to_ms(total_ns),
                "avg_kernel_duration_us": _ns_to_us(float(r["avg_dur_ns"] or 0.0)),
                "pct_of_total_kernel_time": (_safe_div(float(total_ns), float(total_kernel_time_ns)) * 100.0)
                if total_kernel_time_ns
                else 0.0,
            }
        )

    notes: List[str] = []
    if has_text_id and not stable:
        notes.append("NVTX events may reference textId, but StringIds table was not found.")
    if total_kernel_time_ns and mapped_kernel_time_ns:
        notes.append(
            "Attributed {:.1f}% of total kernel time via NVTX→runtime→kernel correlation.".format(
                _safe_div(float(mapped_kernel_time_ns), float(total_kernel_time_ns)) * 100.0
            )
        )
    elif total_kernel_time_ns and not mapped_kernel_time_ns:
        notes.append("No kernels could be attributed to NVTX ranges (missing correlationId/globalTid linkage).")

    coverage = (_safe_div(float(mapped_kernel_time_ns), float(total_kernel_time_ns)) if total_kernel_time_ns else 0.0)
    return {
        "present": True,
        "kernel_table": ktable,
        "runtime_table": rtable,
        "nvtx_table": ntable,
        "ranges": ranges,
        "total_kernel_time_ns": total_kernel_time_ns,
        "mapped_kernel_time_ns": int(mapped_kernel_time_ns),
        "mapped_kernel_time_ms": _ns_to_ms(mapped_kernel_time_ns),
        "coverage_fraction": float(coverage),
        "coverage_pct": float(coverage * 100.0),
        "mapped_kernel_count": int(mapped_kernel_count),
        "notes": notes,
        "sql": {"nvtx_kernel_ranges": sql},
    }


def _pid_expr_for_table(alias: str, info: Any) -> Tuple[Optional[str], Optional[str], str]:
    """
    Return (pid_expr_sql, pid_source_column, note).

    Nsight Systems commonly encodes:
    - globalTid = pid*0x1000000 + tid
    - globalPid = pid*0x1000000
    """

    if info.has("pid"):
        return ("CAST({}.pid AS INT)".format(alias), "pid", "pid column")
    if info.has("processId"):
        return ("CAST({}.processId AS INT)".format(alias), "processId", "processId column")
    if info.has("globalPid"):
        return ("(CAST({}.globalPid / 16777216 AS INT) % 16777216)".format(alias), "globalPid", "decoded from globalPid")
    if info.has("globalTid"):
        return ("(CAST({}.globalTid / 16777216 AS INT) % 16777216)".format(alias), "globalTid", "decoded from globalTid")
    return (None, None, "no pid-like column found")


def kernels_by_pid(
    trace_db: TraceDB, *, top_pids: int = 10, top_kernels_per_pid: int = 10, limit_pids_for_kernel_rows: int = 10
) -> Dict[str, Any]:
    ktable = trace_db.schema.kernel_table
    stable = trace_db.schema.string_table
    if not ktable:
        return {"present": False, "notes": ["No kernel activity table found."], "sql": {}}

    kinfo = trace_db.schema.table(ktable)
    pid_expr, pid_source, _note = _pid_expr_for_table("k", kinfo)
    if pid_expr is None or pid_source is None:
        return {"present": False, "notes": ["PID breakdown unavailable for kernels (no pid/globalPid/globalTid column)."], "sql": {}}

    name_col = "demangledName" if kinfo.has("demangledName") else ("shortName" if kinfo.has("shortName") else None)
    join = ""
    name_expr = "'<unknown>'"
    if name_col:
        name_expr = "k.{c}".format(c=name_col)
        if stable and not trace_db.schema.is_text_column(ktable, name_col):
            join = " JOIN {s} s ON s.id = k.{c} ".format(s=stable, c=name_col)
            name_expr = "s.value"

    device_expr = "k.deviceId" if kinfo.has("deviceId") else "-1"
    total_kernel_time_ns = int(_fetch_one(trace_db.conn, "SELECT SUM(end-start) FROM {t}".format(t=ktable)) or 0)

    where = "k.{col} IS NOT NULL".format(col=pid_source) if pid_source in ("globalPid", "globalTid") else "1=1"

    # PID attribution quality check (heuristic): count how many kernel rows map to PID 0 or
    # very large PIDs. This helps catch exports where PID decoding is not meaningful.
    pid_quality: Dict[str, Any] = {"present": False}
    sql_pid_quality = (
        "SELECT "
        "COUNT(*) AS rows_with_pid, "
        "SUM(CASE WHEN {pid} = 0 THEN 1 ELSE 0 END) AS pid0_rows, "
        "SUM(CASE WHEN {pid} >= 10000000 THEN 1 ELSE 0 END) AS pid_ge_10m_rows "
        "FROM {t} k WHERE {w}"
    ).format(pid=pid_expr, t=ktable, w=where)
    try:
        q = trace_db.conn.execute(sql_pid_quality).fetchone()
        if q and q["rows_with_pid"] is not None:
            rows_with_pid = int(q["rows_with_pid"] or 0)
            pid0_rows = int(q["pid0_rows"] or 0)
            pid_ge_10m_rows = int(q["pid_ge_10m_rows"] or 0)
            pid_quality = {
                "present": True,
                "rows_with_pid": rows_with_pid,
                "pid0_rows": pid0_rows,
                "pid_ge_10m_rows": pid_ge_10m_rows,
                "pid0_fraction": _safe_div(float(pid0_rows), float(rows_with_pid)),
                "pid_ge_10m_fraction": _safe_div(float(pid_ge_10m_rows), float(rows_with_pid)),
            }
    except Exception:
        pid_quality = {"present": False}
    sql_top_pids = (
        "SELECT {pid} AS pid, SUM(k.end-k.start) AS total_ns, COUNT(*) AS kernel_count "
        "FROM {t} k WHERE {w} GROUP BY pid ORDER BY total_ns DESC LIMIT ?"
    ).format(pid=pid_expr, t=ktable, w=where)
    top = trace_db.conn.execute(sql_top_pids, (int(top_pids),)).fetchall()
    pid_rows: List[Dict[str, Any]] = []
    pid_totals: Dict[int, int] = {}
    for r in top:
        pid = int(r["pid"]) if r["pid"] is not None else -1
        ns = int(r["total_ns"] or 0)
        pid_totals[pid] = ns
        pid_rows.append(
            {
                "pid": pid,
                "total_kernel_time_ns": ns,
                "total_kernel_time_ms": _ns_to_ms(ns),
                "kernel_count": int(r["kernel_count"] or 0),
                "pct_of_total_kernel_time": (_safe_div(float(ns), float(total_kernel_time_ns)) * 100.0) if total_kernel_time_ns else 0.0,
            }
        )

    pids = [row["pid"] for row in pid_rows][: int(limit_pids_for_kernel_rows)]
    kernels_rows: List[Dict[str, Any]] = []
    sql_kernels = None
    rows: Sequence[sqlite3.Row] = []
    per_pid_counts: Dict[int, int] = {}
    if pids:
        pid_binds = ",".join(["?"] * len(pids))
        sql_kernels = (
            "SELECT {pid} AS pid, {name} AS kernel_name, {dev} AS device_id, "
            "COUNT(*) AS call_count, SUM(k.end-k.start) AS total_ns, AVG(k.end-k.start) AS avg_ns "
            "FROM {t} k {join} "
            "WHERE ({pid}) IN ({ph}) "
            "GROUP BY pid, kernel_name, device_id "
            "ORDER BY pid, total_ns DESC"
        ).format(pid=pid_expr, name=name_expr, dev=device_expr, t=ktable, join=join, ph=pid_binds)
        rows = trace_db.conn.execute(sql_kernels, tuple(int(x) for x in pids)).fetchall()
        for r in rows:
            pid = int(r["pid"]) if r["pid"] is not None else -1
            per_pid_counts[pid] = per_pid_counts.get(pid, 0) + 1
            if per_pid_counts[pid] > int(top_kernels_per_pid):
                continue
            total_ns = int(r["total_ns"] or 0)
            pid_total_ns = int(pid_totals.get(pid) or 0)
            kernels_rows.append(
                {
                    "pid": pid,
                    "pid_total_kernel_time_ms": _ns_to_ms(pid_total_ns),
                    "pid_pct_of_total_kernel_time": (_safe_div(float(pid_total_ns), float(total_kernel_time_ns)) * 100.0)
                    if total_kernel_time_ns
                    else 0.0,
                    "kernel_name": str(r["kernel_name"]),
                    "device_id": int(r["device_id"]) if r["device_id"] is not None else None,
                    "call_count": int(r["call_count"] or 0),
                    "total_time_ms": _ns_to_ms(total_ns),
                    "avg_duration_us": _ns_to_us(float(r["avg_ns"] or 0.0)),
                    "pct_of_pid_kernel_time": (_safe_div(float(total_ns), float(pid_total_ns)) * 100.0) if pid_total_ns else 0.0,
                    "pct_of_total_kernel_time": (_safe_div(float(total_ns), float(total_kernel_time_ns)) * 100.0)
                    if total_kernel_time_ns
                    else 0.0,
                }
            )

    return {
        "present": True,
        "kernel_table": ktable,
        "pid_source": pid_source,
        "pid_quality": pid_quality,
        "pids": pid_rows,
        "kernels": kernels_rows,
        "notes": [],
        "sql": (
            {"top_pids": sql_top_pids, "kernels": sql_kernels, "pid_quality": sql_pid_quality}
            if sql_kernels
            else {"top_pids": sql_top_pids, "pid_quality": sql_pid_quality}
        ),
    }


def sync_by_pid(trace_db: TraceDB, *, top_pids: int = 10, limit: int = 200) -> Dict[str, Any]:
    rtable = trace_db.schema.runtime_table
    stable = trace_db.schema.string_table
    if not rtable:
        return {"present": False, "notes": ["No runtime API activity table found."], "sql": {}}

    rinfo = trace_db.schema.table(rtable)
    pid_expr, pid_source, _note = _pid_expr_for_table("r", rinfo)
    if pid_expr is None or pid_source is None:
        return {"present": False, "notes": ["PID breakdown unavailable for runtime (no globalTid/globalPid/pid)."], "sql": {}}

    name_col = "nameId" if rinfo.has("nameId") else ("name" if rinfo.has("name") else None)
    if not name_col:
        return {"present": False, "notes": ["Runtime table missing name/nameId."], "sql": {}}

    join = ""
    name_expr = "r.{c}".format(c=name_col)
    if stable and not trace_db.schema.is_text_column(rtable, name_col):
        join = " JOIN {s} s ON s.id = r.{c} ".format(s=stable, c=name_col)
        name_expr = "s.value"

    sync_keywords = [
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaStreamWaitEvent",
        "cudaEventQuery",
    "cuCtxSynchronize",
    "cuStreamSynchronize",
    "cuEventSynchronize",
    "cuStreamWaitEvent",
    ]
    where_parts = ["({expr} LIKE ?)".format(expr=name_expr) for _ in range(len(sync_keywords) + 2)]
    params: List[Any] = ["%{}%".format(k) for k in sync_keywords] + ["%Wait%", "%Synchronize%"]
    w0 = " OR ".join(where_parts)
    where_pid = "r.{col} IS NOT NULL".format(col=pid_source) if pid_source in ("globalPid", "globalTid") else "1=1"

    sql = (
        "SELECT {pid} AS pid, {name} AS api_name, COUNT(*) AS call_count, "
        "SUM(r.end-r.start) AS total_ns, AVG(r.end-r.start) AS avg_ns "
        "FROM {t} r {join} "
        "WHERE ({w0}) AND ({wpid}) "
        "GROUP BY pid, api_name "
        "ORDER BY total_ns DESC "
        "LIMIT ?"
    ).format(pid=pid_expr, name=name_expr, t=rtable, join=join, w0=w0, wpid=where_pid)

    rows = trace_db.conn.execute(sql, tuple(params) + (int(limit),)).fetchall()
    out: List[Dict[str, Any]] = []
    totals_by_pid: Dict[int, float] = {}
    for r in rows:
        pid = int(r["pid"]) if r["pid"] is not None else -1
        total_ns = int(r["total_ns"] or 0)
        totals_by_pid[pid] = totals_by_pid.get(pid, 0.0) + float(total_ns)
        out.append(
            {
                "pid": pid,
                "api_name": str(r["api_name"]),
                "call_count": int(r["call_count"] or 0),
                "total_time_ms": _ns_to_ms(total_ns),
                "avg_duration_us": _ns_to_us(float(r["avg_ns"] or 0.0)),
            }
        )

    pid_totals = sorted(totals_by_pid.items(), key=lambda kv: kv[1], reverse=True)[: int(top_pids)]
    pid_rows = [{"pid": int(pid), "sync_total_time_ms": _ns_to_ms(ns)} for pid, ns in pid_totals]
    return {"present": True, "runtime_table": rtable, "pid_source": pid_source, "pids": pid_rows, "sync_calls": out, "notes": [], "sql": {"sync_by_pid": sql}}


def nvtx_kernel_time_by_range_by_pid(
    trace_db: TraceDB, *, top_pids: int = 10, top_ranges_per_pid: int = 10
) -> Dict[str, Any]:
    """Per-PID variant of NVTX→runtime→kernel attribution (best-effort)."""

    ktable = trace_db.schema.kernel_table
    rtable = trace_db.schema.runtime_table
    ntable = trace_db.schema.nvtx_table
    stable = trace_db.schema.string_table

    if not ktable or not rtable or not ntable:
        return {"present": False, "notes": ["Need kernel + runtime + NVTX tables for per-PID NVTX→kernel attribution."], "sql": {}}

    kinfo = trace_db.schema.table(ktable)
    rinfo = trace_db.schema.table(rtable)
    ninfo = trace_db.schema.table(ntable)

    def pick_col(info: Any, candidates: Sequence[str]) -> Optional[str]:
        for c in candidates:
            if info.has(c):
                return c
        return None

    k_cid = pick_col(kinfo, ["correlationId", "correlationID", "correlation_id"])
    r_cid = pick_col(rinfo, ["correlationId", "correlationID", "correlation_id"])
    if k_cid is None or r_cid is None:
        return {"present": False, "notes": ["Missing correlationId on kernel/runtime tables."], "sql": {}}

    r_gt = pick_col(rinfo, ["globalTid", "globalTID", "global_tid"])
    n_gt = pick_col(ninfo, ["globalTid", "globalTID", "global_tid"])
    if r_gt is None or n_gt is None:
        return {
            "present": False,
            "notes": ["Need runtime.globalTid and NVTX.globalTid to correlate NVTX ranges to kernel launches."],
            "sql": {},
        }

    pid_expr_r, pid_source_r, _ = _pid_expr_for_table("r", rinfo)
    pid_expr_k, pid_source_k, _ = _pid_expr_for_table("k", kinfo)
    if pid_expr_r is None or pid_source_r is None:
        return {"present": False, "notes": ["Runtime PID decode unavailable (need globalTid/globalPid/pid)."], "sql": {}}
    if pid_expr_k is None or pid_source_k is None:
        return {"present": False, "notes": ["Kernel PID decode unavailable (need globalPid/processId/pid)."], "sql": {}}

    if not ninfo.has("end"):
        return {"present": False, "notes": ["NVTX table missing globalTid/end; cannot correlate to runtime threads."], "sql": {}}

    has_text = ninfo.has("text")
    has_text_id = ninfo.has("textId")
    join = ""
    range_expr = "n.text" if has_text else ("n.textId" if has_text_id else "''")
    if has_text_id and stable and not trace_db.schema.is_text_column(ntable, "textId"):
        join = " LEFT JOIN {s} s ON s.id = n.textId ".format(s=stable)
        range_expr = "COALESCE(n.text, s.value)" if has_text else "s.value"

    nvtx_filter = "WHERE n.end IS NOT NULL AND n.end > n.start"
    if ninfo.has("eventType"):
        nvtx_filter += " AND n.eventType IN (59, 60)"

    # Top PIDs by *total kernel time*.
    where_pid_k = "k.{col} IS NOT NULL".format(col=pid_source_k) if pid_source_k in ("globalPid", "globalTid") else "1=1"
    sql_top_pids = (
        "SELECT {pid} AS pid, SUM(k.end-k.start) AS total_ns, COUNT(*) AS kernel_count "
        "FROM {t} k WHERE {w} GROUP BY pid ORDER BY total_ns DESC LIMIT ?"
    ).format(pid=pid_expr_k, t=ktable, w=where_pid_k)
    pid_totals_rows = trace_db.conn.execute(sql_top_pids, (int(top_pids),)).fetchall()
    pid_totals: Dict[int, int] = {}
    pid_kernel_counts: Dict[int, int] = {}
    pids: List[int] = []
    for r in pid_totals_rows:
        pid = int(r["pid"]) if r["pid"] is not None else -1
        ns = int(r["total_ns"] or 0)
        pid_totals[pid] = ns
        pid_kernel_counts[pid] = int(r["kernel_count"] or 0)
        pids.append(pid)

    if not pids:
        return {"present": False, "notes": ["No kernel rows with PID found."], "sql": {"top_pids": sql_top_pids}}

    pid_binds = ",".join(["?"] * len(pids))
    where_pid_r = "r.{col} IS NOT NULL".format(col=pid_source_r) if pid_source_r in ("globalPid", "globalTid") else "1=1"
    sql = """
        WITH runtime AS (
            SELECT
            r.{r_cid} AS correlation_id,
            r.start AS r_start,
            r.end AS r_end,
            r.{r_gt} AS global_tid,
            {pid_expr_r} AS pid
        FROM {rtable} r
        WHERE r.{r_cid} IS NOT NULL AND r.{r_gt} IS NOT NULL AND r.end IS NOT NULL
          AND ({where_pid_r})
    ),
    nvtx AS (
        SELECT
            n.start AS n_start,
            n.end AS n_end,
            n.{n_gt} AS global_tid,
            {range_expr} AS range_name
        FROM {ntable} n
        {join}
        {nvtx_filter}
          AND n.{n_gt} IS NOT NULL
    ),
    mapped AS (
        SELECT
            runtime.pid AS pid,
            (k.end - k.start) AS dur_ns,
            (
                SELECT n2.range_name
                FROM nvtx n2
                WHERE n2.global_tid = runtime.global_tid
                  AND n2.n_start <= runtime.r_start
                  AND n2.n_end >= runtime.r_end
                ORDER BY n2.n_start DESC
                LIMIT 1
            ) AS range_name
        FROM {ktable} k
        JOIN runtime ON runtime.correlation_id = k.{k_cid}
        WHERE k.{k_cid} IS NOT NULL
          AND k.end IS NOT NULL AND k.end > k.start
          AND runtime.pid IN ({ph})
        )
        SELECT
        pid,
        range_name,
        COUNT(*) AS kernel_count,
        SUM(dur_ns) AS total_dur_ns,
        AVG(dur_ns) AS avg_dur_ns
    FROM mapped
    WHERE range_name IS NOT NULL
    GROUP BY pid, range_name
    ORDER BY pid, total_dur_ns DESC;
    """.format(
        rtable=rtable,
        r_cid=r_cid,
        r_gt=r_gt,
        pid_expr_r=pid_expr_r,
        where_pid_r=where_pid_r,
        ntable=ntable,
        n_gt=n_gt,
        range_expr=range_expr,
        join=join,
        nvtx_filter=nvtx_filter,
        ktable=ktable,
        k_cid=k_cid,
        ph=pid_binds,
    )

    rows = trace_db.conn.execute(sql, tuple(int(x) for x in pids)).fetchall()
    by_pid_seen: Dict[int, int] = {}
    ranges: List[Dict[str, Any]] = []
    mapped_by_pid_ns: Dict[int, int] = {}
    for r in rows:
        pid = int(r["pid"]) if r["pid"] is not None else -1
        by_pid_seen[pid] = by_pid_seen.get(pid, 0) + 1
        total_ns = int(r["total_dur_ns"] or 0)
        mapped_by_pid_ns[pid] = mapped_by_pid_ns.get(pid, 0) + total_ns
        if by_pid_seen[pid] <= int(top_ranges_per_pid):
            ranges.append(
                {
                    "pid": pid,
                    "range_name": str(r["range_name"]),
                    "kernel_count": int(r["kernel_count"] or 0),
                    "total_kernel_time_ns": total_ns,
                    "total_kernel_time_ms": _ns_to_ms(total_ns),
                    "avg_kernel_duration_us": _ns_to_us(float(r["avg_dur_ns"] or 0.0)),
                }
            )

    pid_summaries: List[Dict[str, Any]] = []
    for pid in pids:
        total_ns = int(pid_totals.get(pid) or 0)
        mapped_ns = int(mapped_by_pid_ns.get(pid) or 0)
        cov = (_safe_div(float(mapped_ns), float(total_ns)) if total_ns else 0.0)
        pid_summaries.append(
            {
                "pid": pid,
                "pid_total_kernel_time_ms": _ns_to_ms(total_ns),
                "pid_total_kernel_count": int(pid_kernel_counts.get(pid) or 0),
                "pid_attributed_kernel_time_ms": _ns_to_ms(mapped_ns),
                "pid_attribution_coverage_fraction": float(cov),
                "pid_attribution_coverage_pct": float(cov * 100.0),
            }
        )

    return {
        "present": True,
        "kernel_table": ktable,
        "runtime_table": rtable,
        "nvtx_table": ntable,
        "pids": pid_summaries,
        "ranges": ranges,
        "notes": [],
        "sql": {"top_pids": sql_top_pids, "nvtx_kernel_by_pid": sql},
    }


def nvtx_by_pid(trace_db: TraceDB, *, limit: int = 200) -> Dict[str, Any]:
    ntable = trace_db.schema.nvtx_table
    if not ntable:
        return {"present": False, "notes": ["No NVTX table found."], "sql": {}}

    ninfo = trace_db.schema.table(ntable)
    if not ninfo.has("globalTid"):
        return {"present": False, "notes": ["NVTX PID breakdown unavailable (NVTX table missing globalTid)."], "sql": {}}
    if not ninfo.has("end"):
        return {"present": False, "notes": ["NVTX table missing end column."], "sql": {}}

    stable = trace_db.schema.string_table
    has_text = ninfo.has("text")
    has_text_id = ninfo.has("textId")
    range_expr = "n.text" if has_text else ("n.textId" if has_text_id else "''")
    join = ""
    if has_text_id and stable and not trace_db.schema.is_text_column(ntable, "textId"):
        join = " LEFT JOIN {s} s ON s.id = n.textId ".format(s=stable)
        range_expr = "COALESCE(n.text, s.value)" if has_text else "s.value"

    pid_expr = "(CAST(n.globalTid / 16777216 AS INT) % 16777216)"

    nvtx_filter = "WHERE n.end IS NOT NULL AND n.end > n.start"
    if ninfo.has("eventType"):
        nvtx_filter += " AND n.eventType IN (59, 60)"

    sql_nvtx = (
        "SELECT {pid} AS pid, {name} AS range_name, COUNT(*) AS nvtx_count, "
        "SUM(n.end-n.start) AS total_ns "
        "FROM {t} n {join} {f} "
        "GROUP BY pid, range_name "
        "ORDER BY total_ns DESC "
        "LIMIT ?"
    ).format(pid=pid_expr, name=range_expr, t=ntable, join=join, f=nvtx_filter)
    rows = trace_db.conn.execute(sql_nvtx, (int(limit),)).fetchall()
    nvtx_rows: List[Dict[str, Any]] = []
    for r in rows:
        ns = int(r["total_ns"] or 0)
        nvtx_rows.append(
            {
                "pid": int(r["pid"]) if r["pid"] is not None else -1,
                "range_name": str(r["range_name"]),
                "nvtx_count": int(r["nvtx_count"] or 0),
                "nvtx_total_time_ms": _ns_to_ms(ns),
            }
        )

    nvk_by_pid = nvtx_kernel_time_by_range_by_pid(trace_db, top_pids=10, top_ranges_per_pid=20)

    # Merge wall-time and kernel-time attribution on (pid, range_name).
    key_to_row: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for r in nvtx_rows:
        key = (int(r["pid"]), str(r["range_name"]))
        key_to_row[key] = dict(r)
        key_to_row[key].update(
            {"attributed_kernel_time_ms": 0.0, "attributed_kernel_count": 0, "pid_attribution_coverage_pct": None}
        )

    pid_cov: Dict[int, float] = {}
    if nvk_by_pid.get("present"):
        for p in nvk_by_pid.get("pids") or []:
            pid_cov[int(p["pid"])] = float(p.get("pid_attribution_coverage_pct") or 0.0)
        for r in nvk_by_pid.get("ranges") or []:
            key = (int(r["pid"]), str(r["range_name"]))
            row = key_to_row.get(key) or {"pid": int(r["pid"]), "range_name": str(r["range_name"]), "nvtx_count": 0, "nvtx_total_time_ms": 0.0}
            row.update(
                {
                    "attributed_kernel_time_ms": float(r.get("total_kernel_time_ms") or 0.0),
                    "attributed_kernel_count": int(r.get("kernel_count") or 0),
                    "pid_attribution_coverage_pct": pid_cov.get(int(r["pid"])),
                }
            )
            key_to_row[key] = row

    merged_rows = sorted(key_to_row.values(), key=lambda rr: (int(rr.get("pid") or -1), -float(rr.get("attributed_kernel_time_ms") or 0.0), -float(rr.get("nvtx_total_time_ms") or 0.0)))
    return {
        "present": True,
        "nvtx_table": ntable,
        "pid_source": "globalTid",
        "ranges": merged_rows,
        "kernel_time_by_pid": nvk_by_pid,
        "notes": [],
        "sql": {"nvtx_ranges_by_pid": sql_nvtx},
    }


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for k in rows[0].keys():
        if k not in seen:
            fieldnames.append(k)
            seen.add(k)
    for r in rows[1:]:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
