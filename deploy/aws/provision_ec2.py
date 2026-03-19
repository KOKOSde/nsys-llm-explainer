#!/usr/bin/env python3
"""Provision a public EC2 instance and bootstrap nsys-llm-api service."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import stat
import time
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError


SERVICE_PORT = 7860


USER_DATA = """#!/bin/bash
set -euxo pipefail

dnf update -y
dnf install -y python3 python3-pip git

python3 -m venv /opt/nsys-venv
/opt/nsys-venv/bin/python -m pip install --upgrade pip
/opt/nsys-venv/bin/python -m pip install "git+https://github.com/KOKOSde/nsys-llm-explainer.git@v0.3.0#egg=nsys-llm-explainer[api]"

cat >/etc/systemd/system/nsys-llm-api.service <<'UNIT'
[Unit]
Description=nsys-llm-explainer API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
Environment=PORT=7860
ExecStart=/opt/nsys-venv/bin/nsys-llm-api --host 0.0.0.0 --port 7860
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now nsys-llm-api
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Provision EC2 for nsys-llm-explainer API.")
    parser.add_argument("--region", default="us-east-1", help="AWS region.")
    parser.add_argument("--instance-type", default="t3.small", help="EC2 instance type.")
    parser.add_argument("--name-prefix", default="nsys-llm-api", help="Name tag prefix.")
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


def _ensure_security_group(ec2_client: Any, *, vpc_id: str, group_name: str, allow_ssh: bool) -> str:
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
            "FromPort": SERVICE_PORT,
            "ToPort": SERVICE_PORT,
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
        "UserData": USER_DATA,
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
        "api_url": "http://{}:{}/healthz".format(public_ip, SERVICE_PORT) if public_ip else None,
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
