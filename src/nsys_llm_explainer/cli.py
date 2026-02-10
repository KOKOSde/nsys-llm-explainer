import argparse
from pathlib import Path
from typing import List, Optional

from .queries import TraceDB
from .report import analyze, write_artifacts


def _format_table_listing(db: TraceDB, *, max_cols: Optional[int] = None) -> str:
    lines: List[str] = []
    for name in sorted(db.schema.tables.keys()):
        info = db.schema.tables[name]
        # Row count can be expensive, but --print-schema is a diagnostic path.
        try:
            row_count = int(db.conn.execute('SELECT COUNT(*) FROM \"{}\"'.format(name)).fetchone()[0] or 0)
        except Exception:
            row_count = -1

        win = None
        try:
            has_start = "start" in info.columns
            has_end = "end" in info.columns
            if has_start and has_end:
                r = db.conn.execute('SELECT MIN(start), MAX(end) FROM \"{}\"'.format(name)).fetchone()
                if r and r[0] is not None and r[1] is not None:
                    win = (int(r[0]), int(r[1]))
            elif has_start:
                r = db.conn.execute('SELECT MIN(start), MAX(start) FROM \"{}\"'.format(name)).fetchone()
                if r and r[0] is not None and r[1] is not None:
                    win = (int(r[0]), int(r[1]))
        except Exception:
            win = None

        cols = list(info.columns)
        if max_cols is not None and len(cols) > int(max_cols):
            cols = cols[: int(max_cols)] + ["..."]
        cols_with_types = []
        for c in cols:
            if c == "...":
                cols_with_types.append("...")
                continue
            t = info.column_types.get(c, "")
            cols_with_types.append("{}:{}".format(c, t) if t else c)
        meta_parts: List[str] = []
        if row_count >= 0:
            meta_parts.append("rows={}".format(row_count))
        if win:
            meta_parts.append("t=[{},{}]".format(win[0], win[1]))
        meta = (" (" + ", ".join(meta_parts) + ")") if meta_parts else ""
        lines.append("- {}{}: {}".format(name, meta, ", ".join(cols_with_types)))
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nsys-llm-explain",
        description="Turn an Nsight Systems SQLite export into an actionable LLM inference performance report.",
    )
    p.add_argument("trace_sqlite", help="Path to trace.sqlite (from `nsys export --type sqlite ...`).")
    p.add_argument("--out", required=True, help="Output directory (e.g. artifacts/<run_id>/).")
    p.add_argument(
        "--phase-map",
        "--phases-json",
        dest="phase_map",
        default=None,
        help="Optional JSON mapping NVTX range names to phases.",
    )
    p.add_argument(
        "--kernel-limit",
        "--top-kernels",
        dest="kernel_limit",
        type=int,
        default=30,
        help="How many top kernels to include (default: 30).",
    )
    p.add_argument(
        "--no-kernel-percentiles",
        action="store_true",
        help="Skip per-kernel p50/p90 computation (faster for very large traces).",
    )
    p.add_argument(
        "--print-schema",
        action="store_true",
        help="Print detected key tables to stdout.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = TraceDB.open(args.trace_sqlite)
    try:
        if args.print_schema:
            s = db.schema
            print("Detected tables:")
            print("  String table:", s.string_table)
            print("  Kernel table:", s.kernel_table)
            print("  Runtime table:", s.runtime_table)
            print("  NVTX table:", s.nvtx_table)
            print("  Sync table:", s.sync_table)
            print("")
            print("All tables (first columns):")
            print(_format_table_listing(db))
            print("")

        outputs = analyze(
            db,
            phase_map_path=args.phase_map,
            kernel_limit=int(args.kernel_limit),
            compute_kernel_percentiles=not bool(args.no_kernel_percentiles),
        )
        write_artifacts(outputs, out_dir)

        m = outputs.report["metrics"]
        top_k = m["top_kernels"].get("kernels") or []
        top = top_k[0] if top_k else None
        launch = m["launch_storm"]
        idle = m["gpu_idle"].get("devices") or []
        worst = max(idle, key=lambda d: float(d.get("idle_pct_of_window") or 0.0)) if idle else None

        print("Wrote report to:", out_dir / "report.md")
        if top:
            print(
                "Top kernel:",
                top.get("kernel_name"),
                "({:.1f}% of kernel time, {:.1f} ms, {} calls)".format(
                    float(top.get("pct_total_kernel_time") or 0.0),
                    float(top.get("total_time_ms") or 0.0),
                    int(top.get("call_count") or 0),
                ),
            )
        print(
            "Launch storm:",
            "{} launches over {:.3f}s = {:.1f} launches/s; median kernel {:.2f} us".format(
                int(launch.get("total_launches") or 0),
                float(launch.get("window_s") or 0.0),
                float(launch.get("launches_per_s") or 0.0),
                float(launch.get("median_kernel_us") or 0.0),
            ),
        )
        if worst:
            print(
                "GPU idle estimate:",
                "GPU {}: {:.1f}% idle ({:.1f} ms / {:.1f} ms window)".format(
                    worst.get("device_id"),
                    float(worst.get("idle_pct_of_window") or 0.0),
                    float(worst.get("idle_ms") or 0.0),
                    float(worst.get("window_ms") or 0.0),
                ),
            )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

