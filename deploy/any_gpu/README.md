# Any GPU Host Quickstart (Vast / AWS / Personal)

This path is provider-agnostic. It installs the API as a `systemd` service on any Linux GPU VM.

## 1. Prepare host

- Ubuntu/Debian/Amazon Linux VM
- Open inbound TCP port (default `7860`) in your provider firewall/security group
- SSH access with sudo

## 2. Install service

From your VM shell:

```bash
git clone https://github.com/KOKOSde/nsys-llm-explainer.git
cd nsys-llm-explainer

# Optional: set API key (recommended for public endpoints)
export NSYS_API_KEY="change-me"

sudo -E bash deploy/any_gpu/install_service.sh \
  --repo-ref v0.3.2 \
  --port 7860
```

## 3. Verify

```bash
curl -sS http://127.0.0.1:7860/healthz
```

From your laptop (replace `<PUBLIC_IP>`):

```bash
curl -sS http://<PUBLIC_IP>:7860/healthz
```

## 4. Auth behavior

- If `NSYS_API_KEY` is unset: API is public.
- If set: `/v1/analyze/*` requires one of:
  - `x-api-key: <secret>`
  - `Authorization: Bearer <secret>`
- `/` and `/healthz` stay public for liveness checks.

## 5. Upgrade

Re-run installer with a new `--repo-ref` (for example `v0.3.3`).

## 6. Uninstall

```bash
sudo systemctl disable --now nsys-llm-api
sudo rm -f /etc/systemd/system/nsys-llm-api.service /etc/default/nsys-llm-api /usr/local/bin/start_nsys_api.sh
sudo systemctl daemon-reload
```
