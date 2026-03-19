"""Minimal client demo for nsys-llm-explainer API."""

from __future__ import annotations

import argparse
from pathlib import Path

from nsys_llm_explainer.client import NsysExplainerClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call nsys-llm-explainer API from Python client.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="API base URL.")
    parser.add_argument("--input", required=True, help="Path to .sqlite/.db/.json input.")
    parser.add_argument("--out-zip", default="out/nsys_llm_artifacts.zip", help="Path for artifacts zip output.")
    parser.add_argument("--kernel-limit", type=int, default=30, help="Kernel ranking limit.")
    parser.add_argument("--no-markdown", action="store_true", help="Disable markdown in JSON response.")
    return parser


def main() -> int:
    args = _parser().parse_args()
    client = NsysExplainerClient(args.base_url)

    health = client.health()
    print("health:", health)

    result = client.analyze_json(
        args.input,
        kernel_limit=int(args.kernel_limit),
        include_markdown=not bool(args.no_markdown),
    )
    print("summary:", result.get("summary"))

    out_zip = client.analyze_artifacts(
        args.input,
        output_zip_path=args.out_zip,
        kernel_limit=int(args.kernel_limit),
    )
    print("artifacts_zip:", Path(out_zip).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
