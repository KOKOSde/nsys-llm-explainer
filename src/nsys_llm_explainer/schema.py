"""SQLite schema probing utilities for Nsight Systems exports."""

import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: Tuple[str, ...]
    column_types: Mapping[str, str]

    def has(self, col: str) -> bool:
        return col in self.column_types


@dataclass(frozen=True)
class SchemaProbeResult:
    tables: Mapping[str, TableInfo]
    string_table: Optional[str]
    kernel_table: Optional[str]
    runtime_table: Optional[str]
    nvtx_table: Optional[str]
    sync_table: Optional[str]

    def table(self, name: str) -> TableInfo:
        return self.tables[name]

    def is_text_column(self, table: str, col: str) -> bool:
        t = self.tables[table].column_types.get(col, "").lower()
        return ("text" in t) or ("char" in t) or ("clob" in t)


def sqlite_version(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT sqlite_version()").fetchone()
    return str(row[0]) if row and row[0] is not None else "unknown"


def decode_global_pid(global_pid: Optional[int]) -> Optional[int]:
    """Decode Nsight Systems serialized globalPid into pid (best-effort)."""

    if global_pid is None:
        return None
    # Nsight Systems encodes pid/tid as pid*0x1000000 + tid.
    # For globalPid, tid is effectively 0.
    return (int(global_pid) // 0x1000000) % 0x1000000


def list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(r[0]) for r in rows]


def table_info(conn: sqlite3.Connection, table: str) -> TableInfo:
    rows = conn.execute("PRAGMA table_info({})".format(table)).fetchall()
    cols: List[str] = []
    types: Dict[str, str] = {}
    for r in rows:
        c = str(r[1])
        t = str(r[2] or "")
        cols.append(c)
        types[c] = t
    return TableInfo(name=table, columns=tuple(cols), column_types=types)


def _has_cols(info: TableInfo, cols: Sequence[str]) -> bool:
    return all(info.has(c) for c in cols)


def _pick_best_table(infos: Mapping[str, TableInfo], candidates: Sequence[str], required_cols: Sequence[str]) -> Optional[str]:
    for t in candidates:
        info = infos.get(t)
        if info and _has_cols(info, required_cols):
            return t
    return None


def _pick_string_table(infos: Mapping[str, TableInfo]) -> Optional[str]:
    candidates = ["StringIds", "Strings", "StringIdsV2", "StringTable"]
    for t in candidates:
        info = infos.get(t)
        if info and _has_cols(info, ["id", "value"]):
            return t
    for name, info in infos.items():
        if _has_cols(info, ["id", "value"]):
            return name
    return None


def probe_schema(conn: sqlite3.Connection) -> SchemaProbeResult:
    names = list_tables(conn)
    infos: Dict[str, TableInfo] = {n: table_info(conn, n) for n in names}

    string_table = _pick_string_table(infos)

    kernel_table = _pick_best_table(
        infos,
        candidates=[
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
            "CUDA_GPU_KERNEL",
            "GPU_KERNEL",
        ],
        required_cols=["start", "end"],
    )
    if not kernel_table:
        # Fallback: any table that looks like a kernel activity table.
        for name, info in infos.items():
            if "KERNEL" in name.upper() and _has_cols(info, ["start", "end"]):
                kernel_table = name
                break

    runtime_table = _pick_best_table(
        infos,
        candidates=[
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "CUPTI_ACTIVITY_KIND_DRIVER",
            "CUDA_API_RUNTIME",
            "CUDA_RUNTIME",
        ],
        required_cols=["start", "end"],
    )
    if not runtime_table:
        for name, info in infos.items():
            up = name.upper()
            if ("RUNTIME" in up or "DRIVER" in up or "CUDA_API" in up) and _has_cols(info, ["start", "end"]):
                runtime_table = name
                break

    nvtx_table = _pick_best_table(
        infos,
        candidates=[
            "NVTX_EVENTS",
            "NVTX_EVENTS_V2",
            "NVTX",
        ],
        required_cols=["start"],
    )
    if not nvtx_table:
        for name, info in infos.items():
            if "NVTX" in name.upper() and _has_cols(info, ["start"]):
                nvtx_table = name
                break

    sync_table: Optional[str] = None
    if runtime_table:
        rinfo = infos[runtime_table]
        if rinfo.has("nameId") or rinfo.has("name"):
            sync_table = runtime_table

    return SchemaProbeResult(
        tables=infos,
        string_table=string_table,
        kernel_table=kernel_table,
        runtime_table=runtime_table,
        nvtx_table=nvtx_table,
        sync_table=sync_table,
    )


def decode_global_tid(global_tid: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    """Decode Nsight Systems serialized globalTid into (pid, tid)."""
    if global_tid is None:
        return (None, None)
    g = int(global_tid)
    tid = g % 0x1000000
    pid = (g // 0x1000000) % 0x1000000
    return (pid, tid)


def decode_global_pid(global_pid: Optional[int]) -> Optional[int]:
    """Decode Nsight Systems serialized globalPid into pid."""
    if global_pid is None:
        return None
    return (int(global_pid) // 0x1000000) % 0x1000000

