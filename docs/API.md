# nsys-llm-explainer API

The API service wraps the same analysis pipeline used by the CLI and dashboard.

## Install

```bash
python3 -m pip install -e .[api,client]
```

## Run service

```bash
nsys-llm-api --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl -s http://127.0.0.1:8080/healthz | jq
```

## Analyze to JSON

```bash
curl -sS -X POST \
  -F "file=@trace.sqlite" \
  -F "kernel_limit=50" \
  -F "include_markdown=true" \
  http://127.0.0.1:8080/v1/analyze/json | jq
```

Optional phase map:

```bash
curl -sS -X POST \
  -F "file=@trace.sqlite" \
  -F "phase_map=@phases.json" \
  http://127.0.0.1:8080/v1/analyze/json | jq
```

## Analyze to artifacts ZIP

```bash
curl -sS -X POST \
  -F "file=@trace.sqlite" \
  -F "kernel_limit=50" \
  http://127.0.0.1:8080/v1/analyze/artifacts \
  -o nsys_llm_artifacts.zip
```

## Python client

```python
from nsys_llm_explainer.client import NsysExplainerClient

client = NsysExplainerClient("http://127.0.0.1:8080")
print(client.health())

result = client.analyze_json("trace.sqlite", kernel_limit=50, include_markdown=True)
print(result["summary"])

client.analyze_artifacts("trace.sqlite", output_zip_path="out/nsys_llm_artifacts.zip")
```

## Notes

- File uploads support `.sqlite`, `.db`, and `.json`.
- For `.json` uploads, the API returns parsed report payloads and best-effort markdown rendering.
- Default upload size limit is 2 GiB (`MAX_UPLOAD_BYTES` in `src/nsys_llm_explainer/api.py`).
