"""Trace-derived queries/metrics from an Nsight Systems SQLite export."""

import contextlib
import csv
import json
import math
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
    }

    # Best-effort time unit guess (Nsight Systems exports are typically nanoseconds).
    time_unit_assumed = "ns"
    time_unit_guess = "unknown"
    if trace_db.schema.kernel_table:
        row = trace_db.conn.execute(
            "SELECT MIN(start) AS s, MAX(end) AS e FROM {t}".format(t=trace_db.schema.kernel_table)
        ).fetchone()
        if row and row["s"] is not None and row["e"] is not None:
            window = int(row["e"]) - int(row["s"])
            # If window is large, it is almost certainly nanoseconds.
            if window >= 1_000_000_000:  # >= 1s if ns; >= ~11 days if us
                time_unit_guess = "ns"
            elif window >= 1_000_000:  # >= 1ms if ns; >= 1s if us
                time_unit_guess = "ns_likely"

    return {
        "sqlite_version": sqlite_version(trace_db.conn),
        "path": str(trace_db.path),
        "string_table": trace_db.schema.string_table,
        "kernel_table": trace_db.schema.kernel_table,
        "runtime_table": trace_db.schema.runtime_table,
        "nvtx_table": trace_db.schema.nvtx_table,
        "sync_table": trace_db.schema.sync_table,
        "kernel_pid_source": kernel_pid_source,
        "runtime_pid_source": runtime_pid_source,
        "nvtx_pid_source": nvtx_pid_source,
        "timestamp_unit_assumed": time_unit_assumed,
        "timestamp_unit_guess": time_unit_guess,
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
    if pids:
        placeholders = ",".join(["?"] * len(pids))
        sql_kernels = (
            "SELECT {pid} AS pid, {name} AS kernel_name, {dev} AS device_id, "
            "COUNT(*) AS call_count, SUM(k.end-k.start) AS total_ns, AVG(k.end-k.start) AS avg_ns "
            "FROM {t} k {join} "
            "WHERE ({pid}) IN ({ph}) "
            "GROUP BY pid, kernel_name, device_id "
            "ORDER BY pid, total_ns DESC"
        ).format(pid=pid_expr, name=name_expr, dev=device_expr, t=ktable, join=join, ph=placeholders)
        rows = trace_db.conn.execute(sql_kernels, tuple(int(x) for x in pids)).fetchall()
        per_pid_counts: Dict[int, int] = {}
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
        "pids": pid_rows,
        "kernels": kernels_rows,
        "notes": [],
        "sql": {"top_pids": sql_top_pids, "kernels": sql_kernels} if sql_kernels else {"top_pids": sql_top_pids},
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

    placeholders = ",".join(["?"] * len(pids))
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
        ph=placeholders,
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

