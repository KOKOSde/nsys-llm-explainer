#!/usr/bin/env python3
"""Provision a public EC2 instance and bootstrap nsys-llm-api service."""

import argparse
import datetime as dt
import json
import stat
import time
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError


DEFAULT_SERVICE_PORT = 7860
DEFAULT_REPO_URL = "https://github.com/KOKOSde/nsys-llm-explainer.git"
DEFAULT_REPO_REF = "v0.3.0"


USER_DATA_TEMPLATE = """#!/bin/bash
set -euxo pipefail

dnf install -y python3 python3-pip python3-setuptools git

rm -rf /opt/nsys-llm-explainer /opt/nsys-venv
git clone --depth 1 --branch {repo_ref} {repo_url} /opt/nsys-llm-explainer

python3 -m venv /opt/nsys-venv
/opt/nsys-venv/bin/python -m pip install --upgrade pip setuptools wheel
/opt/nsys-venv/bin/python -m pip install "/opt/nsys-llm-explainer[api]"

cat >/usr/local/bin/start_nsys_api.sh <<'SCRIPT'
#!/bin/bash
set -euo pipefail
cd /opt/nsys-llm-explainer
exec /opt/nsys-venv/bin/python -m nsys_llm_explainer.api --host 0.0.0.0 --port {service_port}
SCRIPT
chmod +x /usr/local/bin/start_nsys_api.sh

cat >/etc/default/nsys-llm-api <<'ENV'
NSYS_API_KEY={api_key}
ENV

cat >/etc/systemd/system/nsys-llm-api.service <<'UNIT'
[Unit]
Description=nsys-llm-explainer API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
EnvironmentFile=-/etc/default/nsys-llm-api
Environment=PORT={service_port}
ExecStart=/usr/local/bin/start_nsys_api.sh
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now nsys-llm-api
"""


def _render_user_data(*, repo_url: str, repo_ref: str, service_port: int, api_key: str) -> str:
    if "\n" in str(api_key) or "\r" in str(api_key):
        raise ValueError("API key cannot contain newline characters.")
    return USER_DATA_TEMPLATE.format(
        repo_url=str(repo_url),
        repo_ref=str(repo_ref),
        service_port=int(service_port),
        api_key=str(api_key),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision EC2 for nsys-llm-explainer API.")
    parser.add_argument("--region", default="us-east-1", help="AWS region.")
    parser.add_argument("--instance-type", default="t3.small", help="EC2 instance type.")
    parser.add_argument("--service-port", type=int, default=DEFAULT_SERVICE_PORT, help="API service port.")
    parser.add_argument("--name-prefix", default="nsys-llm-api", help="Name tag prefix.")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Git URL to clone on the instance.")
    parser.add_argument("--repo-ref", default=DEFAULT_REPO_REF, help="Git branch/tag/sha to deploy.")
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key. If set, /v1 endpoints require x-api-key or Bearer token.",
    )
    parser.add_argument("--allow-ssh", action="store_true", help="Allow inbound SSH from 0.0.0.0/0.")
    parser.add_argument("--create-key-pair", action="store_true", help="Create a new key pair and write PEM locally.")
    parser.add_argument(
        "--output-json",
        default="deploy/aws/ec2_deploy_output.json",
        help="Path to deployment output JSON.",
    )
    return parser


def _latest_al2023_ami(ssm_client: Any, ec2_client: Any) -> str:
    try:
        param = ssm_client.get_parameter(Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64")
        return str(param["Parameter"]["Value"])
    except ClientError:
        images = ec2_client.describe_images(
            Owners=["amazon"],
            Filters=[
                {"Name": "name", "Values": ["al2023-ami-2023*-x86_64"]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": ["x86_64"]},
                {"Name": "root-device-type", "Values": ["ebs"]},
            ],
        ).get("Images", [])
        if not images:
            raise RuntimeError("Could not resolve an Amazon Linux 2023 AMI.")
        latest = sorted(images, key=lambda img: str(img.get("CreationDate") or ""))[-1]
        return str(latest["ImageId"])


def _default_vpc_and_subnet(ec2_client: Any) -> Dict[str, str]:
    vpcs = ec2_client.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}]).get("Vpcs", [])
    if not vpcs:
        raise RuntimeError("No default VPC found in this region.")
    vpc_id = str(vpcs[0]["VpcId"])
    subnets = ec2_client.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]).get("Subnets", [])
    if not subnets:
        raise RuntimeError("No subnet found in default VPC.")
    # Prefer subnets that map public IP on launch, fallback to first.
    subnets_sorted = sorted(subnets, key=lambda s: (not bool(s.get("MapPublicIpOnLaunch")), s.get("AvailabilityZone", "")))
    subnet_id = str(subnets_sorted[0]["SubnetId"])
    return {"vpc_id": vpc_id, "subnet_id": subnet_id}


def _ensure_security_group(
    ec2_client: Any,
    *,
    vpc_id: str,
    group_name: str,
    allow_ssh: bool,
    service_port: int,
) -> str:
    existing = ec2_client.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [group_name]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]
    ).get("SecurityGroups", [])
    if existing:
        sg_id = str(existing[0]["GroupId"])
    else:
        sg_id = str(
            ec2_client.create_security_group(
                GroupName=group_name,
                Description="Security group for nsys-llm-explainer API",
                VpcId=vpc_id,
            )["GroupId"]
        )

    permissions = [
        {
            "IpProtocol": "tcp",
            "FromPort": int(service_port),
            "ToPort": int(service_port),
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Public API"}],
        }
    ]
    if allow_ssh:
        permissions.append(
            {
                "IpProtocol": "tcp",
                "FromPort": 22,
                "ToPort": 22,
                "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH (open)"}],
            }
        )
    try:
        ec2_client.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=permissions)
    except ClientError as exc:
        if "InvalidPermission.Duplicate" not in str(exc):
            raise
    return sg_id


def _maybe_create_key_pair(ec2_client: Any, *, name: str, out_dir: Path) -> Optional[Path]:
    try:
        response = ec2_client.create_key_pair(KeyName=name, KeyType="rsa")
    except ClientError as exc:
        if "InvalidKeyPair.Duplicate" in str(exc):
            return None
        raise
    pem = str(response["KeyMaterial"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pem_path = out_dir / "{}.pem".format(name)
    pem_path.write_text(pem, encoding="utf-8")
    pem_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return pem_path


def main() -> int:
    args = _parser().parse_args()
    region = str(args.region)
    service_port = int(args.service_port)
    if service_port < 1 or service_port > 65535:
        raise SystemExit("--service-port must be in range 1..65535")

    session = boto3.Session(region_name=region)
    ec2_client = session.client("ec2")
    ssm_client = session.client("ssm")

    timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    name_prefix = str(args.name_prefix)
    name = "{}-{}".format(name_prefix, timestamp)
    group_name = "{}-sg".format(name_prefix)
    key_name = "{}-key-{}".format(name_prefix, timestamp)

    network = _default_vpc_and_subnet(ec2_client)
    ami_id = _latest_al2023_ami(ssm_client, ec2_client)
    sg_id = _ensure_security_group(
        ec2_client,
        vpc_id=network["vpc_id"],
        group_name=group_name,
        allow_ssh=bool(args.allow_ssh),
        service_port=service_port,
    )

    key_pair_path = None
    key_name_for_instance = None
    if bool(args.create_key_pair):
        key_pair_path = _maybe_create_key_pair(ec2_client, name=key_name, out_dir=Path("deploy/aws"))
        key_name_for_instance = key_name

    run_args: Dict[str, Any] = {
        "ImageId": ami_id,
        "InstanceType": str(args.instance_type),
        "MinCount": 1,
        "MaxCount": 1,
        "NetworkInterfaces": [
            {
                "DeviceIndex": 0,
                "AssociatePublicIpAddress": True,
                "SubnetId": network["subnet_id"],
                "Groups": [sg_id],
            }
        ],
        "UserData": _render_user_data(
            repo_url=str(args.repo_url),
            repo_ref=str(args.repo_ref),
            service_port=service_port,
            api_key=str(args.api_key),
        ),
    }
    if key_name_for_instance:
        run_args["KeyName"] = key_name_for_instance

    instances = ec2_client.run_instances(**run_args)["Instances"]
    instance_id = str(instances[0]["InstanceId"])
    print("Launched instance:", instance_id)

    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])

    desc = ec2_client.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
    public_ip = str(desc.get("PublicIpAddress") or "")
    public_dns = str(desc.get("PublicDnsName") or "")
    az = str(desc.get("Placement", {}).get("AvailabilityZone") or "")

    print("Waiting 90 seconds for bootstrap...")
    time.sleep(90)

    output = {
        "region": region,
        "instance_id": instance_id,
        "instance_type": str(args.instance_type),
        "ami_id": ami_id,
        "availability_zone": az,
        "vpc_id": network["vpc_id"],
        "subnet_id": network["subnet_id"],
        "security_group_id": sg_id,
        "security_group_name": group_name,
        "public_ip": public_ip,
        "public_dns": public_dns,
        "api_url": "http://{}:{}/healthz".format(public_ip, service_port) if public_ip else None,
        "api_auth_mode": "api_key" if str(args.api_key).strip() else "public",
        "key_pair_name": key_name_for_instance,
        "key_pair_path": str(key_pair_path) if key_pair_path else None,
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
