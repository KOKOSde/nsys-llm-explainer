"""Python client for nsys-llm-explainer API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import requests


class NsysExplainerClient:
    """Thin client around the nsys-llm-explainer HTTP API."""

    def __init__(self, base_url: str, *, timeout_s: int = 300) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = int(timeout_s)

    def health(self) -> Dict[str, Any]:
        response = requests.get(self.base_url + "/healthz", timeout=self.timeout_s)
        response.raise_for_status()
        return dict(response.json())

    def analyze_json(
        self,
        input_path: str,
        *,
        kernel_limit: int = 30,
        include_markdown: bool = True,
        phase_map_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError("Input path not found: {}".format(path))

        files = {"file": (path.name, path.read_bytes())}
        if phase_map_path:
            phase_path = Path(phase_map_path)
            if not phase_path.exists():
                raise FileNotFoundError("Phase map path not found: {}".format(phase_path))
            files["phase_map"] = (phase_path.name, phase_path.read_bytes())
        data = {"kernel_limit": int(kernel_limit), "include_markdown": bool(include_markdown)}

        response = requests.post(
            self.base_url + "/v1/analyze/json",
            files=files,
            data=data,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return dict(response.json())

    def analyze_artifacts(
        self,
        input_path: str,
        *,
        output_zip_path: str,
        kernel_limit: int = 30,
        phase_map_path: Optional[str] = None,
    ) -> Path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError("Input path not found: {}".format(path))

        files = {"file": (path.name, path.read_bytes())}
        if phase_map_path:
            phase_path = Path(phase_map_path)
            if not phase_path.exists():
                raise FileNotFoundError("Phase map path not found: {}".format(phase_path))
            files["phase_map"] = (phase_path.name, phase_path.read_bytes())
        data = {"kernel_limit": int(kernel_limit)}

        response = requests.post(
            self.base_url + "/v1/analyze/artifacts",
            files=files,
            data=data,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        out_path = Path(output_zip_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(response.content)
        return out_path
