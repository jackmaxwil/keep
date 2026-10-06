# SkyPilot GLM-5.2 break-glass runbook

This runbook is restricted to AWS account `246813579024`, `us-west-2`, one
on-demand `p5.48xlarge`, and the authenticated campaign descriptor. Every
helper invocation first verifies an explicitly named R&D SSO profile. It never
buys a Capacity Block, requests Spot, or launches EC2 directly.

Set the exact authorities:

```bash
cd /Users/jack.mazac/Developer/keep
export AWS_PROFILE=YOUR_NAMED_RND_PROFILE
export CAMPAIGN_DESCRIPTOR=/absolute/path/campaign-descriptor-v2.json
export SKY_BIN=/absolute/path/to/pinned/skypilot-0.13.0/bin/sky
export SKYPILOT_CONFIG=/absolute/path/rendered-skypilot-config.yaml
```

Inspect the Managed Job queue and tagged AWS resources. The first command is
the pinned-bin equivalent of `sky jobs queue --output json`:

```bash
AWS_PROFILE="$AWS_PROFILE" "$SKY_BIN" jobs queue \
  --config "$SKYPILOT_CONFIG" --output json
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh status
```

Inspect controller logs and worker logs:

```bash
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh logs
```

Inspect the worker through SSM, including `systemctl`, `journalctl`, spend
status, and heartbeat:

```bash
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh inspect-worker
```

Request the normal boundary-checkpoint protocol before cancellation:

```bash
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh graceful-stop
```

After confirming recent S3 checkpoint and heartbeat progress, perform a bounded
SkyPilot graceful cancellation:

```bash
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh cancel
```

Authenticate `CAMPAIGN_DRAINED.json`, the campaign phase ledger, and
`GPU_SPEND_LEDGER.jsonl`:

```bash
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh verify
```

If the worker disappears, let the same Managed Job recover it. If the job is
terminal, create a fresh 12-hour submission descriptor with the same immutable
campaign identity and same run ID, stage and rehearse that descriptor, then
resume. The `resume` action calls the normal guarded submission path; the spend
ledger grants only the remaining approved balance and never resets the
24-hour authorization.

Finally audit instances, EBS volumes, and Elastic IPs for orphan resources:

```bash
aws/glm52-gpu/scripts/sky_campaign_break_glass.sh audit-orphans
```

Use SkyPilot recovery or the guarded same-run submission only. Never use a raw
EC2 launch path, purchase a Capacity Block, change region, enable Spot, or
extend the approved GPU budget from this runbook.
