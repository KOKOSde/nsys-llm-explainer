#!/usr/bin/env python3
"""Upsert Cloudflare DNS record (A/CNAME) using token from env."""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request


BASE = "https://api.cloudflare.com/client/v4"


def _request(method: str, path: str, token: str, payload=None):
    url = BASE + path
    data = None
    headers = {
        "Authorization": "Bearer {}".format(token),
        "Content-Type": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers=headers, method=method.upper())
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    if not parsed.get("success"):
        raise RuntimeError("Cloudflare API error: {}".format(parsed))
    return parsed.get("result")


def _zone_id(token: str, zone_name: str) -> str:
    query = urllib.parse.urlencode({"name": zone_name})
    result = _request("GET", "/zones?{}".format(query), token)
    if not result:
        raise RuntimeError("Zone not found: {}".format(zone_name))
    return str(result[0]["id"])


def _find_record(token: str, zone_id: str, rtype: str, name: str):
    query = urllib.parse.urlencode({"type": rtype, "name": name})
    result = _request("GET", "/zones/{}/dns_records?{}".format(zone_id, query), token)
    return result[0] if result else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Upsert Cloudflare DNS record.")
    parser.add_argument("--zone", required=True, help="Zone name, e.g. kokosdev.com")
    parser.add_argument("--name", required=True, help="Record name, e.g. api.kokosdev.com")
    parser.add_argument("--type", default="A", choices=["A", "CNAME"], help="Record type")
    parser.add_argument("--content", required=True, help="Record target value")
    parser.add_argument("--ttl", type=int, default=1, help="TTL seconds, or 1 for auto")
    parser.add_argument("--proxied", action="store_true", help="Enable Cloudflare proxy")
    args = parser.parse_args()

    token = os.getenv("CF_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set CF_API_TOKEN env var before running.")

    zone_id = _zone_id(token, str(args.zone))
    payload = {
        "type": str(args.type),
        "name": str(args.name),
        "content": str(args.content),
        "ttl": int(args.ttl),
        "proxied": bool(args.proxied),
    }

    existing = _find_record(token, zone_id, str(args.type), str(args.name))
    if existing:
        record_id = str(existing["id"])
        result = _request("PUT", "/zones/{}/dns_records/{}".format(zone_id, record_id), token, payload)
        action = "updated"
    else:
        result = _request("POST", "/zones/{}/dns_records".format(zone_id), token, payload)
        action = "created"

    print(json.dumps({"action": action, "record": result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
