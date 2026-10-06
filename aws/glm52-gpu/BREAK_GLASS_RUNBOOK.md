> **SUPERSEDED — MUST NOT BE USED FOR THE CURRENT GLM-5.2 CAMPAIGN.**
> The binding decision is
> [`docs/superpowers/decisions/2026-07-30-glm52-member-account-lifecycle-controls.md`](../../docs/superpowers/decisions/2026-07-30-glm52-member-account-lifecycle-controls.md).
> The approved replacement is
> [`aws/glm52-gpu/SKYPILOT_BREAK_GLASS.md`](SKYPILOT_BREAK_GLASS.md), executed
> through `aws/glm52-gpu/scripts/sky_campaign_break_glass.sh`. This historical
> runbook violates the decision because it is Capacity-Block based and routes
> replacement through the legacy Capacity Reservation controller instead of
> SkyPilot-owned on-demand provisioning.

## Current approved break-glass path

Use the exact active production descriptor read back by Phase Ten; never guess
its path. The current operator sequence is:

```bash
cd /Users/jack.mazac/Developer/keep
export AWS_PROFILE=keep-gpu
export FULL_RUN_WORK
FULL_RUN_WORK=$(cat /tmp/glm52-full-run-current)
test -d "$FULL_RUN_WORK"
export CAMPAIGN_DESCRIPTOR="$FULL_RUN_WORK/production/campaign-descriptor-v2.json"
export SKY_BIN=/Users/jack.mazac/.local/share/keep/skypilot-0.13.0/bin/sky
export SKYPILOT_CONFIG=/Users/jack.mazac/.local/share/keep/skypilot-0.13.0/server-config.yaml
test -f "$CAMPAIGN_DESCRIPTOR"

aws/glm52-gpu/scripts/sky_campaign_break_glass.sh status
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh logs
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh inspect-worker
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh graceful-stop
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh verify
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh audit-orphans
```

Request `graceful-stop` and authenticate checkpoint evidence before `cancel`,
unless immediate termination is required to prevent unauthorized spend. The
legacy `campaign_break_glass.py` below validates a Capacity Reservation and
re-invokes the Capacity-Block controller; it is not an approved current-campaign
fallback and **must not be executed**.

# GLM-5.2 Capacity Block break-glass runbook

Use this only for campaign `glm52-teich-20260717` in account `246813579024`,
region `us-west-2`. The controller and its delivery re-invocation are
idempotent. Never use a raw `RunInstances` loop and never purchase replacement
capacity from this runbook.

```bash
cd /Users/jack.mazac/Developer/keep
export AWS_REGION=us-west-2
export RUN_ID=glm52-teich-20260717
export BUCKET=keep-glm52-models-246813579024-us-west-2
export PREFIX=campaigns/$RUN_ID
```

## 1. Inspect the complete watchdog report

This checks the reservation, running instances, both EC2 status checks, SSM,
the systemd service, current campaign phase, CloudWatch alarms, and recent S3
heartbeat/checkpoint markers.

```bash
aws/glm52-gpu/scripts/watch_campaign.py --region "$AWS_REGION"
```

An exit status of `2` means the JSON report contains an actionable alert. A
`drained` status is terminal and safe.

## 2. Inspect controller logs and delivery failures

```bash
aws logs tail /aws/lambda/keep-glm52-capacity-block-controller \
  --since 1h --region "$AWS_REGION"
aws cloudwatch describe-alarms --alarm-name-prefix keep-glm52 \
  --region "$AWS_REGION" \
  --query 'MetricAlarms[].[AlarmName,StateValue,StateReason]' --output table
aws sqs get-queue-url --queue-name keep-glm52-controller-dlq \
  --region "$AWS_REGION"
```

If the DLQ alarm is active, inspect without deleting:

```bash
QUEUE_URL=$(aws sqs get-queue-url --queue-name keep-glm52-controller-dlq \
  --region "$AWS_REGION" --query QueueUrl --output text)
aws sqs receive-message --queue-url "$QUEUE_URL" --max-number-of-messages 10 \
  --visibility-timeout 0 --attribute-names All --message-attribute-names All \
  --region "$AWS_REGION" --output json
```

## 3. Inspect EC2, user data, SSM, and systemd

```bash
INSTANCE_ID=$(aws ec2 describe-instances --region "$AWS_REGION" \
  --filters "Name=tag:campaign-run-id,Values=$RUN_ID" \
    'Name=instance-state-name,Values=pending,running,stopping,stopped' \
  --query 'Reservations[].Instances[] | [0].InstanceId' --output text)
aws ec2 describe-instance-status --include-all-instances \
  --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --output json
aws ssm describe-instance-information --region "$AWS_REGION" \
  --filters "Key=InstanceIds,Values=$INSTANCE_ID" --output json
```

When SSM reports `Online`, submit one read-only inspection command:

```bash
COMMAND_ID=$(aws ssm send-command --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
  --comment 'GLM-5.2 break-glass read-only inspection' \
  --parameters '{"commands":["systemctl status keep-glm52-campaign.service --no-pager || true","journalctl -u keep-glm52-campaign.service -n 100 --no-pager || true","tail -n 100 /var/log/keep-bootstrap.log || true","tail -n 100 /var/log/cloud-init-output.log || true"]}' \
  --query Command.CommandId --output text)
aws ssm wait command-executed --region "$AWS_REGION" \
  --command-id "$COMMAND_ID" --instance-id "$INSTANCE_ID" || true
aws ssm get-command-invocation --region "$AWS_REGION" \
  --command-id "$COMMAND_ID" --instance-id "$INSTANCE_ID" --output json
```

## 4. Re-invoke a missed delivery event safely

First run the dry preview. The helper refuses unless the exact reservation ID,
instance type, AZ, end time, and active state still match the signed campaign
descriptor.

```bash
aws/glm52-gpu/scripts/campaign_break_glass.py delivery --region "$AWS_REGION"
aws/glm52-gpu/scripts/campaign_break_glass.py delivery --region "$AWS_REGION" --execute
```

The controller returns `already-running` if an active campaign instance exists.
Duplicate invocations use the same client token and cannot launch a second
instance.

## 5. Request an immediate graceful stop

```bash
aws/glm52-gpu/scripts/campaign_break_glass.py graceful-stop --region "$AWS_REGION"
aws/glm52-gpu/scripts/campaign_break_glass.py graceful-stop \
  --region "$AWS_REGION" --execute
```

This invokes the same controller path as the AWS expiration warning. It writes
`/run/keep-glm52/STOP` and sends `SIGTERM` to the campaign service. Do not force
terminate until the newest checkpoint and ledger are visible in S3.

## 6. Verify the newest durable checkpoint and ledger

```bash
aws s3api list-objects-v2 --bucket "$BUCKET" --prefix "$PREFIX/" \
  --region "$AWS_REGION" \
  --query 'reverse(sort_by(Contents,&LastModified))[:30].[LastModified,Size,Key]' \
  --output table
aws s3 cp "s3://$BUCKET/$PREFIX/ledger/latest.json" - \
  --region "$AWS_REGION" --only-show-errors
aws s3 cp "s3://$BUCKET/$PREFIX/teacher-checkpoints/latest.json" - \
  --region "$AWS_REGION" --only-show-errors || true
aws s3 cp "s3://$BUCKET/$PREFIX/training-checkpoints/latest.json" - \
  --region "$AWS_REGION" --only-show-errors || true
aws s3 cp "s3://$BUCKET/$PREFIX/monitor/heartbeat.json" - \
  --region "$AWS_REGION" --only-show-errors
```

The selected `latest.json` marker is authoritative only when the immutable
objects it names exist and hash correctly. Use the campaign’s normal restore
path for full chain validation; do not infer completion from filenames alone.

## 7. Replace a terminated instance inside the same Capacity Block

Only do this after the graceful-stop and S3 verification steps above. Confirm
the reservation is still `active` and that enough time remains to resume useful
work:

```bash
aws ec2 describe-capacity-reservations --region "$AWS_REGION" \
  --capacity-reservation-ids cr-0b628a28bb3adbb96 \
  --query 'CapacityReservations[0].[State,StartDate,EndDate,InstanceType,AvailabilityZone]' \
  --output table
aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-terminated --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
aws/glm52-gpu/scripts/campaign_break_glass.py delivery \
  --region "$AWS_REGION" --execute
```

After termination, the controller derives the next deterministic replacement
generation and launches against the existing Capacity Reservation ID. It does
not call `PurchaseCapacityBlock`, request Spot, or create another reservation.
The replacement restores the ledger and checkpoint trees from the same S3
campaign prefix.
