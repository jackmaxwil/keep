"""AWS Lambda entrypoint; bundle with glm52_capacity_block_controller.py."""

from __future__ import annotations

import json
import os

import boto3

from glm52_capacity_block_controller import handle_capacity_block_event


def lambda_handler(event, _context):  # noqa: ANN001
    parameter_name = os.environ["CAMPAIGN_DESCRIPTOR_PARAMETER"]
    ssm = boto3.client("ssm")
    descriptor = json.loads(
        ssm.get_parameter(Name=parameter_name, WithDecryption=False)["Parameter"][
            "Value"
        ]
    )
    return handle_capacity_block_event(
        event,
        descriptor,
        ec2=boto3.client("ec2"),
        ssm=ssm,
    )
