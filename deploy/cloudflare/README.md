# Cloudflare DNS Quickstart

Use this helper to point a custom domain to your API endpoint without hardcoding secrets.

## 1. Create token

Cloudflare Dashboard:

1. `My Profile` -> `API Tokens` -> `Create Token`
2. Custom token permissions:
   - `Zone:Read`
   - `DNS:Edit`
3. Scope it to one zone (recommended).

## 2. Export token (runtime only)

```bash
export CF_API_TOKEN="your-token"
```

## 3. Create/update record

Point `api.kokosdev.com` to a VM public IP:

```bash
python3 deploy/cloudflare/upsert_dns_record.py \
  --zone kokosdev.com \
  --name api.kokosdev.com \
  --type A \
  --content 100.31.238.27 \
  --proxied
```

Point `api.kokosdev.com` to another hostname:

```bash
python3 deploy/cloudflare/upsert_dns_record.py \
  --zone kokosdev.com \
  --name api.kokosdev.com \
  --type CNAME \
  --content some-target.example.com \
  --proxied
```

No token is written to repo files.
