import json
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path

# Allow tests to run without installing the package.
sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from nsys_llm_explainer.queries import (
    TraceDB,
    detect_launch_storm,
    estimate_gpu_idle_gaps,
    find_sync_events,
    get_top_kernels,
    kernels_by_pid,
    nvtx_breakdown,
)


def _mk_global_tid(pid: int, tid: int) -> int:
    return int(pid) * 0x1000000 + int(tid)


def _mk_global_pid(pid: int) -> int:
    return int(pid) * 0x1000000


class TestSyntheticSQLiteFixtures(unittest.TestCase):
    def test_kernel_table_only(self) -> None:
        """Fixture 1: kernel table only (no runtime, no nvtx)."""

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "trace.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.executescript(
                    """
                    CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                        start INT NOT NULL,
                        end INT NOT NULL,
                        deviceId INT NOT NULL,
                        contextId INT NOT NULL,
                        streamId INT NOT NULL,
                        demangledName INT NOT NULL
                    );
                    """
                )
                conn.executemany("INSERT INTO StringIds(id, value) VALUES(?, ?)", [(1, "kernelA"), (2, "kernelB")])
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,contextId,streamId,demangledName) VALUES(?,?,?,?,?,?)",
                    [
                        (0, 1_000_000, 0, 0, 0, 1),
                        (2_000_000, 3_000_000, 0, 0, 0, 1),
                        (10_000_000, 11_000_000, 0, 0, 0, 2),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            db = TraceDB.open(db_path)
            try:
                top = get_top_kernels(db, limit=10, compute_percentiles=True)
                self.assertTrue(top["kernels"])
                idle = estimate_gpu_idle_gaps(db, top_n_gaps=10)
                self.assertTrue(idle["devices"])
                # runtime/nvtx absent should not crash
                self.assertEqual(find_sync_events(db)["table"], None)
                self.assertEqual(nvtx_breakdown(db)["table"], None)
            finally:
                db.close()

    def test_runtime_sync_events_present(self) -> None:
        """Fixture 2: runtime API sync events present."""

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "trace.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.executescript(
                    """
                    CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                        start INT NOT NULL,
                        end INT NOT NULL,
                        nameId INT NOT NULL,
                        globalTid INT
                    );
                    """
                )
                conn.executemany("INSERT INTO StringIds(id, value) VALUES(?, ?)", [(10, "cudaDeviceSynchronize")])
                gt = _mk_global_tid(1001, 7)
                conn.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(start,end,nameId,globalTid) VALUES(?,?,?,?)",
                    (0, 1_000_000, 10, gt),
                )
                conn.commit()
            finally:
                conn.close()

            db = TraceDB.open(db_path)
            try:
                sync = find_sync_events(db)
                names = [r["api_name"] for r in sync["sync_calls"]]
                self.assertIn("cudaDeviceSynchronize", names)
            finally:
                db.close()

    def test_kernels_by_pid_two_pids(self) -> None:
        """Fixture 3: two PIDs with different kernels; verify kernel PID grouping works."""

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "trace.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.executescript(
                    """
                    CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                        start INT NOT NULL,
                        end INT NOT NULL,
                        deviceId INT NOT NULL,
                        contextId INT NOT NULL,
                        streamId INT NOT NULL,
                        globalPid INT,
                        demangledName INT NOT NULL
                    );
                    """
                )
                conn.executemany(
                    "INSERT INTO StringIds(id, value) VALUES(?, ?)",
                    [(1, "kernelP1"), (2, "kernelP2")],
                )
                pid1 = 111
                pid2 = 222
                gp1 = _mk_global_pid(pid1)
                gp2 = _mk_global_pid(pid2)
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,contextId,streamId,globalPid,demangledName) VALUES(?,?,?,?,?,?,?)",
                    [
                        (0, 10_000_000, 0, 0, 0, gp1, 1),
                        (0, 1_000_000, 0, 0, 0, gp2, 2),
                        (2_000_000, 3_000_000, 0, 0, 0, gp2, 2),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            db = TraceDB.open(db_path)
            try:
                per = kernels_by_pid(db, top_pids=5, top_kernels_per_pid=10, limit_pids_for_kernel_rows=5)
                self.assertTrue(per["present"])
                top_pids = [int(r["pid"]) for r in per["pids"]]
                self.assertIn(pid1, top_pids)
                self.assertIn(pid2, top_pids)
                self.assertEqual(int(per["pids"][0]["pid"]), pid1)
            finally:
                db.close()

    def test_pid_breakdown_and_nvtx_coverage_warning(self) -> None:
        """Fixture 4: two PIDs, NVTX only on one PID -> low coverage warning + by-PID tables."""

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            db_path = td / "trace.sqlite"
            out_dir = td / "out"

            conn = sqlite3.connect(str(db_path))
            try:
                conn.executescript(
                    """
                    CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                        start INT NOT NULL,
                        end INT NOT NULL,
                        deviceId INT NOT NULL,
                        contextId INT NOT NULL,
                        streamId INT NOT NULL,
                        globalPid INT,
                        correlationId INT,
                        demangledName INT NOT NULL
                    );
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                        start INT NOT NULL,
                        end INT NOT NULL,
                        nameId INT NOT NULL,
                        globalTid INT,
                        correlationId INT
                    );
                    CREATE TABLE NVTX_EVENTS(
                        start INT NOT NULL,
                        end INT,
                        eventType INT NOT NULL,
                        text TEXT,
                        textId INT,
                        globalTid INT
                    );
                    """
                )
                conn.executemany(
                    "INSERT INTO StringIds(id, value) VALUES(?, ?)",
                    [
                        (1, "kernelA"),
                        (2, "kernelB"),
                        (9, "cudaLaunchKernel"),
                        (10, "cudaDeviceSynchronize"),
                        (11, "cudaStreamSynchronize"),
                        (20, "decode"),
                    ],
                )

                pid1 = 100
                pid2 = 200
                gp1 = _mk_global_pid(pid1)
                gp2 = _mk_global_pid(pid2)
                gt1 = _mk_global_tid(pid1, 7)
                gt2 = _mk_global_tid(pid2, 7)

                # Kernels: pid2 dominates.
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,contextId,streamId,globalPid,correlationId,demangledName) VALUES(?,?,?,?,?,?,?,?)",
                    [
                        (0, 1_000_000, 0, 0, 0, gp1, 111, 1),
                        (0, 9_000_000, 0, 0, 0, gp2, 222, 2),
                    ],
                )

                # Runtime launch rows (for correlation mapping) + sync-like rows per PID (correlationId NULL).
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(start,end,nameId,globalTid,correlationId) VALUES(?,?,?,?,?)",
                    [
                        (100, 200, 9, gt1, 111),
                        (100, 200, 9, gt2, 222),
                        (300, 300_300, 10, gt1, None),
                        (400, 401_000, 11, gt2, None),
                    ],
                )

                # NVTX only for pid1.
                conn.execute(
                    "INSERT INTO NVTX_EVENTS(start,end,eventType,text,textId,globalTid) VALUES(?,?,?,?,?,?)",
                    (0, 10_000_000, 59, None, 20, gt1),
                )
                conn.commit()
            finally:
                conn.close()

            db = TraceDB.open(db_path)
            try:
                from nsys_llm_explainer.report import analyze, write_artifacts

                phases_path = td / "phases.json"
                phases_path.write_text(json.dumps({"decode": ["decode"]}))

                outputs = analyze(
                    db,
                    phase_map_path=str(phases_path),
                    kernel_limit=10,
                    compute_kernel_percentiles=False,
                    compute_nvtx_kernel_map=True,
                    nvtx_coverage_warn_threshold=0.70,
                )
                self.assertTrue(outputs.report.get("warnings"))
                # Ensure coverage fields exist in JSON.
                nvk = outputs.report["metrics"]["nvtx_kernel_time"]
                self.assertIn("coverage_fraction", nvk)
                self.assertIn("coverage_pct", nvk)

                by_pid = outputs.report["metrics"]["by_pid"]
                self.assertTrue(by_pid["kernels"]["present"])
                pids = [int(r["pid"]) for r in by_pid["kernels"]["pids"]]
                self.assertIn(pid1, pids)
                self.assertIn(pid2, pids)

                self.assertTrue(by_pid["sync"]["present"])
                sync_pids = {int(r["pid"]) for r in (by_pid["sync"]["sync_calls"] or [])}
                self.assertIn(pid1, sync_pids)
                self.assertIn(pid2, sync_pids)

                write_artifacts(outputs, out_dir)
                self.assertTrue((out_dir / "tables" / "kernels_by_pid.csv").exists())
                self.assertTrue((out_dir / "tables" / "sync_by_pid.csv").exists())
                self.assertTrue((out_dir / "tables" / "nvtx_by_pid.csv").exists())
            finally:
                db.close()

    def test_pid_plausibility_warning_all_zero(self) -> None:
        """PID plausibility: decoded PIDs all 0 should warn."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            db_path = td / "trace.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.executescript(
                    """
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                        start INT NOT NULL,
                        end INT NOT NULL,
                        deviceId INT NOT NULL,
                        contextId INT NOT NULL,
                        streamId INT NOT NULL,
                        globalPid INT,
                        demangledName INT NOT NULL
                    );
                    """
                )
                # globalPid=0 -> pid decodes to 0
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,contextId,streamId,globalPid,demangledName) VALUES(?,?,?,?,?,?,?)",
                    [
                        (0, 1_000_000, 0, 0, 0, 0, 1),
                        (2_000_000, 3_000_000, 0, 0, 0, 0, 1),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            db = TraceDB.open(db_path)
            try:
                from nsys_llm_explainer.report import analyze

                outputs = analyze(db, phase_map_path=None, kernel_limit=10, compute_kernel_percentiles=False, compute_nvtx_kernel_map=False)
                warns = outputs.report.get("warnings") or []
                self.assertTrue(any("PID attribution may be unavailable/ambiguous" in str(w) for w in warns))
                pid_meta = outputs.report["metrics"].get("pid_attribution") or {}
                self.assertEqual(int(pid_meta.get("kernel_pid_count") or 0), 1)
            finally:
                db.close()

    def test_pid_plausibility_warning_kernel_single_runtime_multi(self) -> None:
        """PID plausibility: kernel shows 1 PID but runtime shows >1 PID should warn."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            db_path = td / "trace.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.executescript(
                    """
                    CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                        start INT NOT NULL,
                        end INT NOT NULL,
                        deviceId INT NOT NULL,
                        contextId INT NOT NULL,
                        streamId INT NOT NULL,
                        globalPid INT,
                        demangledName INT NOT NULL
                    );
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                        start INT NOT NULL,
                        end INT NOT NULL,
                        nameId INT NOT NULL,
                        globalTid INT
                    );
                    """
                )
                conn.executemany("INSERT INTO StringIds(id,value) VALUES(?,?)", [(10, "cudaDeviceSynchronize"), (1, "k")])
                # kernels only for pid 123
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,contextId,streamId,globalPid,demangledName) VALUES(?,?,?,?,?,?,?)",
                    [(0, 1_000_000, 0, 0, 0, _mk_global_pid(123), 1)],
                )
                # runtime sync rows for two pids 123 and 456
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(start,end,nameId,globalTid) VALUES(?,?,?,?)",
                    [
                        (0, 1000, 10, _mk_global_tid(123, 7)),
                        (0, 1000, 10, _mk_global_tid(456, 7)),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            db = TraceDB.open(db_path)
            try:
                from nsys_llm_explainer.report import analyze

                outputs = analyze(db, phase_map_path=None, kernel_limit=10, compute_kernel_percentiles=False, compute_nvtx_kernel_map=False)
                warns = outputs.report.get("warnings") or []
                self.assertTrue(any("PID attribution may be unavailable/ambiguous" in str(w) for w in warns))
            finally:
                db.close()

    def test_launch_storm_classification_thresholds(self) -> None:
        """Launch storm: many tiny kernels in short window should classify True."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "trace.sqlite"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.executescript(
                    """
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                        start INT NOT NULL,
                        end INT NOT NULL,
                        deviceId INT NOT NULL,
                        contextId INT NOT NULL,
                        streamId INT NOT NULL,
                        globalPid INT,
                        demangledName INT NOT NULL
                    );
                    """
                )
                gp = _mk_global_pid(1234)
                # 200 kernels of 1us duration, spaced 2us apart -> window ~ 400us => 500k launches/s
                rows = []
                t = 0
                for _ in range(200):
                    rows.append((t, t + 1_000, 0, 0, 0, gp, 1))
                    t += 2_000
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,contextId,streamId,globalPid,demangledName) VALUES(?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
            finally:
                conn.close()

            db = TraceDB.open(db_path)
            try:
                ls = detect_launch_storm(None, trace_db=db)
                self.assertIsNotNone(ls.get("is_launch_storm"))
                self.assertTrue(bool(ls.get("is_launch_storm")))
                # per-PID path also classifies
                from nsys_llm_explainer.queries import per_pid_breakdown

                pp = per_pid_breakdown(db, top_pids=5, kernel_limit=5)
                self.assertTrue(pp.get("present"))
                pid0 = pp["pids"][0]
                self.assertTrue(bool((pid0.get("launch_storm") or {}).get("is_launch_storm")))
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
