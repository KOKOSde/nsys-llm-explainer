# Deployment Kit

This directory contains production-oriented deployment assets for the `nsys-llm-explainer` API service.

The intended runtime entrypoint is:

```bash
python -m nsys_llm_explainer.api
```

## Contents

- `Dockerfile.api`: lean container image for API/service deployments.
- `docker-compose.api.yml`: local production-like compose stack with volume mounts.
- `aws/README.md`: AWS ECS/Fargate and EC2 quickstart.
- `gcp/README.md`: Google Cloud Run quickstart.

## Recommended runtime model

- Keep the container stateless.
- Mount trace inputs read-only.
- Mount an output directory for generated reports, artifacts, and logs.
- Configure the service through environment variables rather than baked-in paths.

## Security checklist

- Run as a non-root user.
- Do not bake secrets into the image.
- Prefer read-only input mounts.
- Keep temporary files inside a dedicated writable volume.
- Expose only the API port required by the platform.
- Use short-lived credentials from the cloud provider identity system when possible.

## Observability checklist

- Emit structured startup logs.
- Log the resolved input path, output path, and request correlation ID when available.
- Export request latency, error counts, and queue depth if the API is stateful.
- Surface memory usage and CPU saturation in platform metrics.
- Keep a small health endpoint (`/healthz` in this service).

## Suggested next step

Pair this deployment kit with a public Hugging Face Space and a short technical write-up that shows:

- a live upload path,
- a trace-to-insight demo,
- a cloud deployment recipe,
- and a reproducible benchmark or comparison result.
