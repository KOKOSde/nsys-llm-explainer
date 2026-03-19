"""HTTP API for nsys-llm-explainer."""

from __future__ import annotations

import argparse
import hmac
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .queries import TraceDB
from .report import AnalysisOutputs, analyze, render_markdown, write_artifacts


MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
UNPROTECTED_PATHS = {"/", "/healthz"}


def _safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return int(default)
    return max(minimum, min(maximum, parsed))


def _configured_api_key() -> Optional[str]:
    key = str(os.getenv("NSYS_API_KEY", "")).strip()
    return key or None


def _extract_api_key(headers: Mapping[str, str]) -> Optional[str]:
    key = str(headers.get("x-api-key") or "").strip()
    if key:
        return key
    auth = str(headers.get("authorization") or "").strip()
    lower = auth.lower()
    if lower.startswith("bearer "):
        token = auth[7:].strip()
        if token:
            return token
    return None


def _summarize_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics") or {}
    top_kernel = ((metrics.get("top_kernels") or {}).get("kernels") or [None])[0]
    top_barrier = ((metrics.get("barriers") or {}).get("barriers") or [None])[0]
    top_nccl = ((metrics.get("nccl") or {}).get("ops") or [None])[0]
    findings = list(report.get("findings") or [])
    severities = {"high": 0, "medium": 0, "low": 0}
    for finding in findings:
        level = str((finding or {}).get("severity") or "").lower()
        if level in severities:
            severities[level] += 1

    return {
        "tool_version": (report.get("tool") or {}).get("version"),
        "warnings_count": len(report.get("warnings") or []),
        "findings_count": len(findings),
        "severity_counts": severities,
        "top_kernel": {
            "name": (top_kernel or {}).get("kernel_name"),
            "total_time_ms": (top_kernel or {}).get("total_time_ms"),
            "pct_kernel_time": (top_kernel or {}).get("pct_total_kernel_time"),
        }
        if top_kernel
        else None,
        "top_barrier": {
            "name": (top_barrier or {}).get("api_name"),
            "kind": (top_barrier or {}).get("barrier_kind"),
            "total_time_ms": (top_barrier or {}).get("total_time_ms"),
        }
        if top_barrier
        else None,
        "top_nccl_op": {
            "name": (top_nccl or {}).get("op_name"),
            "total_time_ms": (top_nccl or {}).get("total_time_ms"),
            "compute_overlap_pct": (top_nccl or {}).get("compute_overlap_pct"),
        }
        if top_nccl
        else None,
    }


def _zip_dir(root: Path) -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(root)))
    data.seek(0)
    return data.read()


def _ensure_allowed_file_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in (".sqlite", ".db", ".json"):
        raise HTTPException(status_code=400, detail="Only .sqlite, .db, or .json files are supported.")
    return suffix


async def _read_upload(upload: UploadFile) -> bytes:
    blob = await upload.read()
    if not blob:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the 2 GiB service limit.")
    return blob


def _analyze_sqlite(sqlite_path: Path, *, phase_map_path: Optional[Path], kernel_limit: int) -> AnalysisOutputs:
    db = TraceDB.open(sqlite_path)
    try:
        return analyze(
            db,
            phase_map_path=str(phase_map_path) if phase_map_path else None,
            kernel_limit=int(kernel_limit),
            compute_kernel_percentiles=True,
            compute_nvtx_kernel_map=True,
        )
    finally:
        db.close()


def _parse_report_json(blob: bytes) -> Dict[str, Any]:
    try:
        parsed = json.loads(blob.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON input: {}".format(exc)) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="Input JSON must be an object.")
    return parsed


app = FastAPI(
    title="nsys-llm-explainer API",
    version=__version__,
    description="Upload Nsight Systems SQLite exports and receive report JSON/Markdown or artifact bundles.",
)


@app.middleware("http")
async def _api_key_guard(request: Any, call_next: Any) -> Any:
    required_key = _configured_api_key()
    if not required_key:
        return await call_next(request)
    if str(request.method).upper() == "OPTIONS":
        return await call_next(request)

    path = str(request.url.path or "/")
    if path in UNPROTECTED_PATHS:
        return await call_next(request)

    supplied_key = _extract_api_key(request.headers)
    if not supplied_key or not hmac.compare_digest(str(supplied_key), str(required_key)):
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized. Provide x-api-key or Authorization: Bearer <key>."},
        )
    return await call_next(request)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "nsys-llm-explainer",
        "version": __version__,
        "auth_mode": "api_key" if _configured_api_key() else "public",
        "endpoints": ["/healthz", "/v1/analyze/json", "/v1/analyze/artifacts"],
    }


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "nsys-llm-explainer",
        "version": __version__,
        "auth_enabled": bool(_configured_api_key()),
    }


@app.post("/v1/analyze/json")
async def analyze_json(
    file: UploadFile = File(...),
    phase_map: Optional[UploadFile] = File(None),
    kernel_limit: int = Form(30),
    include_markdown: bool = Form(True),
) -> JSONResponse:
    file_name = str(file.filename or "upload.sqlite")
    suffix = _ensure_allowed_file_suffix(file_name)
    kernel_limit = _safe_int(kernel_limit, default=30, minimum=1, maximum=500)

    with tempfile.TemporaryDirectory(prefix="nsys_llm_api_") as td:
        tmp_dir = Path(td)
        input_path = tmp_dir / ("input" + suffix)
        input_blob = await _read_upload(file)
        input_path.write_bytes(input_blob)

        phase_map_path: Optional[Path] = None
        if phase_map is not None:
            phase_map_blob = await _read_upload(phase_map)
            phase_map_path = tmp_dir / "phase_map.json"
            phase_map_path.write_bytes(phase_map_blob)

        if suffix == ".json":
            report = _parse_report_json(input_blob)
            markdown = None
            if include_markdown:
                try:
                    markdown = render_markdown(report)
                except Exception:
                    markdown = None
        else:
            try:
                outputs = _analyze_sqlite(input_path, phase_map_path=phase_map_path, kernel_limit=kernel_limit)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail="Analysis failed: {}".format(exc)) from exc
            report = dict(outputs.report)
            markdown = outputs.markdown if include_markdown else None

    payload: Dict[str, Any] = {"summary": _summarize_report(report), "report": report}
    if include_markdown:
        payload["markdown"] = markdown
    return JSONResponse(payload)


@app.post("/v1/analyze/artifacts")
async def analyze_artifacts(
    file: UploadFile = File(...),
    phase_map: Optional[UploadFile] = File(None),
    kernel_limit: int = Form(30),
) -> StreamingResponse:
    file_name = str(file.filename or "upload.sqlite")
    suffix = _ensure_allowed_file_suffix(file_name)
    kernel_limit = _safe_int(kernel_limit, default=30, minimum=1, maximum=500)

    with tempfile.TemporaryDirectory(prefix="nsys_llm_artifacts_") as td:
        tmp_dir = Path(td)
        input_path = tmp_dir / ("input" + suffix)
        input_blob = await _read_upload(file)
        input_path.write_bytes(input_blob)

        phase_map_path: Optional[Path] = None
        if phase_map is not None:
            phase_map_blob = await _read_upload(phase_map)
            phase_map_path = tmp_dir / "phase_map.json"
            phase_map_path.write_bytes(phase_map_blob)

        out_dir = tmp_dir / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)

        if suffix == ".json":
            report = _parse_report_json(input_blob)
            (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            try:
                (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
            except Exception:
                (out_dir / "report.md").write_text(
                    "# Nsight Systems LLM Hotspot Report\n\nInput JSON could not be rendered with current markdown formatter.",
                    encoding="utf-8",
                )
        else:
            try:
                outputs = _analyze_sqlite(input_path, phase_map_path=phase_map_path, kernel_limit=kernel_limit)
            except Exception as exc:
                raise HTTPException(status_code=500, detail="Analysis failed: {}".format(exc)) from exc
            write_artifacts(outputs, out_dir)

        zip_blob = _zip_dir(out_dir)

    response_name = "nsys_llm_artifacts.zip"
    return StreamingResponse(
        io.BytesIO(zip_blob),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="{}"'.format(response_name)},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nsys-llm-api",
        description="Run nsys-llm-explainer HTTP API service.",
    )
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Host to bind (default: 0.0.0.0).")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8080")),
        help="Port to bind (default: 8080 or env PORT).",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development.")
    return parser


def main(argv: Optional[Tuple[str, ...]] = None) -> int:
    args = _parser().parse_args(list(argv) if argv else None)
    try:
        import uvicorn
    except Exception as exc:
        raise SystemExit("uvicorn is required. Install with `pip install -e .[api]`. ({})".format(exc))

    uvicorn.run("nsys_llm_explainer.api:app", host=str(args.host), port=int(args.port), reload=bool(args.reload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
