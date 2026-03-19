# AWS Quickstart

This guide assumes you are deploying the `nsys-llm-explainer` API container built from `deploy/Dockerfile.api`.

## Local smoke test

Build the image:

```bash
docker build -f deploy/Dockerfile.api -t nsys-llm-explainer:api .
```

Run it locally:

```bash
docker run --rm -p 7860:7860 \
  -e PORT=7860 \
  -e TRACE_INPUT_DIR=/data/inbox \
  -e TRACE_OUTPUT_DIR=/data/outbox \
  -v "$PWD/data/inbox:/data/inbox:ro" \
  -v "$PWD/data/outbox:/data/outbox" \
  nsys-llm-explainer:api
```

## Automated EC2 provisioning

This repo now includes:

- `deploy/aws/provision_ec2.py`

It launches an EC2 instance, creates/uses a security group, bootstraps `nsys-llm-api` via systemd, and writes deployment output JSON.

Example:

```bash
python3 deploy/aws/provision_ec2.py \
  --region us-east-1 \
  --instance-type t3.small \
  --name-prefix nsys-llm-api \
  --service-port 7860 \
  --repo-ref v0.3.1 \
  --output-json deploy/aws/ec2_deploy_output.json
```

Optional:

- `--allow-ssh` to open port 22
- `--create-key-pair` to create and save a PEM key under `deploy/aws/`
- `--api-key <secret>` to require `x-api-key`/Bearer token on all endpoints except `/` and `/healthz`

Minimum IAM actions typically required:

- `ec2:DescribeVpcs`, `ec2:DescribeSubnets`, `ec2:DescribeSecurityGroups`
- `ec2:CreateSecurityGroup`, `ec2:AuthorizeSecurityGroupIngress`
- `ec2:RunInstances`, `ec2:DescribeInstances`, `ec2:DescribeInstanceStatus`
- Optional: `ec2:CreateKeyPair`, `ec2:GetConsoleOutput`, `ec2:TerminateInstances`
- Optional AMI resolution path: `ssm:GetParameter` (falls back to `ec2:DescribeImages`)

## EC2 quickstart

Use EC2 when you want the simplest reproducible production path:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker

docker pull nsys-llm-explainer:api
docker run -d --name nsys-llm-explainer \
  --restart unless-stopped \
  -p 7860:7860 \
  -e PORT=7860 \
  -e TRACE_INPUT_DIR=/data/inbox \
  -e TRACE_OUTPUT_DIR=/data/outbox \
  -v /srv/nsys/inbox:/data/inbox:ro \
  -v /srv/nsys/outbox:/data/outbox \
  nsys-llm-explainer:api
```

Practical hardening:

- Put `/srv/nsys/inbox` on an encrypted or access-controlled volume.
- Restrict inbound access with a security group or reverse proxy.
- Mount only the traces you intend the service to process.
- Use a systemd unit or a small wrapper script if you want automated restarts and log rotation.

## ECS/Fargate quickstart

Use ECS/Fargate when you want a managed container runtime with low operational overhead.

1. Push the image to Amazon ECR.
1. Create an ECS task definition that uses the image and exposes container port `7860`.
1. Set environment variables:
   - `PORT=7860`
   - `TRACE_INPUT_DIR=/data/inbox`
   - `TRACE_OUTPUT_DIR=/data/outbox`
1. Attach persistent storage if the API needs to retain outputs beyond task lifetime.
1. Put the service behind an Application Load Balancer or another ingress layer.

Recommended deployment notes:

- Use an IAM task role instead of baking AWS credentials into the container.
- Keep trace inputs in S3 if the API can stream them or stage them on startup.
- Send logs to CloudWatch.
- If the API is CPU-heavy during analysis, size tasks with enough memory headroom for pandas and Plotly.

Minimal ECR flow:

```bash
aws ecr create-repository --repository-name nsys-llm-explainer
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag nsys-llm-explainer:api <account>.dkr.ecr.<region>.amazonaws.com/nsys-llm-explainer:api
docker push <account>.dkr.ecr.<region>.amazonaws.com/nsys-llm-explainer:api
```

## What to show in an application

- A trace upload flow that returns a useful report in one request.
- A container deployment story with ECR plus ECS or EC2.
- A short note on how you handled statelessness, trace storage, and logs.
