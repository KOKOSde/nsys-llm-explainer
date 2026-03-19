# GCP Quickstart

This guide assumes the `nsys-llm-explainer` API container built from `deploy/Dockerfile.api`.

## Cloud Run quickstart

Build locally:

```bash
docker build -f deploy/Dockerfile.api -t nsys-llm-explainer:api .
```

Tag and push to Artifact Registry:

```bash
gcloud auth configure-docker <region>-docker.pkg.dev
docker tag nsys-llm-explainer:api <region>-docker.pkg.dev/<project>/<repo>/nsys-llm-explainer:api
docker push <region>-docker.pkg.dev/<project>/<repo>/nsys-llm-explainer:api
```

Deploy to Cloud Run:

```bash
gcloud run deploy nsys-llm-explainer \
  --image <region>-docker.pkg.dev/<project>/<repo>/nsys-llm-explainer:api \
  --region <region> \
  --platform managed \
  --port 7860 \
  --set-env-vars PORT=7860,TRACE_INPUT_DIR=/data/inbox,TRACE_OUTPUT_DIR=/data/outbox \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated
```

## Recommended Cloud Run pattern

- Store traces in Cloud Storage and sync them into the container only when needed.
- Keep outputs in Cloud Storage or stream them back to the client.
- Use a service account with the minimum permissions needed for object storage and logging.
- Set a higher timeout if analysis requests are large.
- Use concurrency conservatively if the API does CPU-bound post-processing.

## Optional load path

If you want a more controlled workflow, stage trace files from GCS into the container at startup and point the API at `/data/inbox`.

Example environment variables:

```bash
TRACE_INPUT_DIR=/data/inbox
TRACE_OUTPUT_DIR=/data/outbox
PORT=7860
```

## What to show in an application

- A fully reproducible container deploy on Cloud Run.
- A story about secure, least-privilege service accounts.
- A demo that makes trace analysis accessible from a browser without local setup.
