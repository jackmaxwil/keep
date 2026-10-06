> **SUPERSEDED — MUST NOT BE USED FOR THE CURRENT GLM-5.2 CAMPAIGN.**
> The binding decision is
> [`docs/superpowers/decisions/2026-07-30-glm52-member-account-lifecycle-controls.md`](../../docs/superpowers/decisions/2026-07-30-glm52-member-account-lifecycle-controls.md).
> The approved replacement is SkyPilot-owned provisioning through the guarded
> `aws/glm52-gpu/scripts/submit_sky_campaign.sh --production` entrypoint.
> This file is retained for history, but its raw EC2 launch, Spot, `g6e`,
> alternate-worker-shape, and alternate-region paths violate the decision's
> one on-demand `p5.48xlarge` in `us-west-2` rule.

# KEEP GLM-5.2 GPU teacher-gen stack

Templatized, reusable AWS infrastructure for running the **bf16 source-teacher
forward** (recording the reference "answer key" logits the compressed GLM-5.2
model is trained against) on NVIDIA GPUs via MLX's CUDA backend, pulling the
model **fast from S3**.

- **Account:** `246813579024` (research-sandbox), region `us-west-2`.
- **Cost model:** no GPU cost is incurred until a worker starts. Retained S3
  objects, the small SkyPilot controller, CloudWatch, SQS/SNS, Lambda, and
  checksum operations can still incur comparatively small non-GPU charges and
  must be included in the final cost audit.
- **Why CloudFormation (not CDK):** no CDK toolchain is needed — this deploys with
  the AWS CLI already in use, and matches the account's existing CFN stacks.

## Current production path and AWS profile

The approved production path is the on-demand, AWS-only SkyPilot Managed Job,
submitted only through the guarded
`aws/glm52-gpu/scripts/submit_sky_campaign.sh --production` entrypoint. Spot
and the raw `launch_instance.sh` workflow later in this file are
**forbidden for the current campaign** and retained only as historical
interfaces.

This Mac also runs another project work against a different AWS account. Use the
explicit `keep-gpu` profile for every GLM-5.2 campaign command:

```bash
aws login --profile keep-gpu
export AWS_PROFILE=keep-gpu
./scripts/assert_rnd_aws_account.sh
```

The guard must print `246813579024`. Never run bare `aws login`, rely on
`default`, or globally set one AWS profile for both projects. See
[`AWS_ACCOUNT_PROFILES.md`](AWS_ACCOUNT_PROFILES.md) for the complete
two-account setup, per-agent rules, SkyPilot API-server behavior, and local
port-collision recovery.

## Durable SkyPilot and the pre-spend integrity gate

Install the pinned control plane outside the model environment:

```bash
export AWS_PROFILE=keep-gpu
./scripts/assert_rnd_aws_account.sh
./scripts/install_skypilot_control_plane.sh
```

The default durable location is
`~/.local/share/keep/skypilot-0.13.0`. Its
`control-environment.json` authenticates the Python, SkyPilot executable,
version, API-server port, queue-manager port, and AWS account. The queue
manager uses port `50012`; this avoids the local `50011` collision that caused
the earlier API-server startup failure.

Render and load the dedicated AWS-only server configuration before starting
the loopback API server:

```bash
export GLM52_SKY_RUN_ID=glm52-sky-20260724
export GLM52_SKY_BUCKET=keep-glm52-models-246813579024-us-west-2
export SKYPILOT_GLOBAL_CONFIG="$SKYPILOT_CONTROL_VENV/server-config.yaml"
export SKYPILOT_API_SERVER_ENDPOINT=http://127.0.0.1:46580
sed \
  -e "s/__JOBS_BUCKET__/$GLM52_SKY_BUCKET/g" \
  -e "s/__RUN_ID__/$GLM52_SKY_RUN_ID/g" \
  skypilot/skypilot-config.yaml.in > "$SKYPILOT_GLOBAL_CONFIG"
"$SKYPILOT_CONTROL_VENV/bin/sky" api stop
"$SKYPILOT_CONTROL_VENV/bin/sky" api start --host 127.0.0.1
"$SKYPILOT_CONTROL_VENV/bin/sky" check aws -v \
  --config "$SKYPILOT_GLOBAL_CONFIG"
```

The check must report `account=246813579024` and that other clouds are disabled
by `allowed_clouds`. Keep `AWS_PROFILE=keep-gpu`,
`SKYPILOT_GLOBAL_CONFIG`, and `SKYPILOT_API_SERVER_ENDPOINT` exported for all
campaign SkyPilot commands. This avoids changing the shared
`~/.sky/config.yaml` used by other projects.

No H100 job may start until the production S3 inventory and full-object
checksum authority pass:

```bash
./scripts/build_production_s3_inventory.py \
  --profile keep-gpu --region us-west-2 \
  --bucket keep-glm52-models-246813579024-us-west-2 \
  --run-id "$RUN_ID" --output "$INVENTORY"

./scripts/run_s3_batch_checksum_audit.py \
  --profile keep-gpu --region us-west-2 \
  --inventory "$INVENTORY" --output "$CHECKSUM_JOB"

# Run after the S3 Batch job reaches Complete.
./scripts/collect_s3_batch_checksum_report.py \
  --profile keep-gpu --region us-west-2 \
  --job-marker "$CHECKSUM_JOB" --inventory "$INVENTORY" \
  --output "$CHECKSUM_AUTHORITY"

# Preview, then remove only stale unpublished source uploads whose completed
# objects are already covered by the exact checksum authority.
./scripts/cleanup_stale_source_multipart_uploads.py \
  --profile keep-gpu --region us-west-2 \
  --inventory "$INVENTORY" --checksum-authority "$CHECKSUM_AUTHORITY" \
  --initiated-before "$RESTORE_STARTED_AT" \
  --output "$MULTIPART_CLEANUP_DRY_RUN"
./scripts/cleanup_stale_source_multipart_uploads.py \
  --profile keep-gpu --region us-west-2 \
  --inventory "$INVENTORY" --checksum-authority "$CHECKSUM_AUTHORITY" \
  --initiated-before "$RESTORE_STARTED_AT" --apply \
  --output "$MULTIPART_CLEANUP_APPLIED"

./scripts/audit_s3_campaign_artifacts.py \
  --profile keep-gpu --region us-west-2 \
  --inventory "$INVENTORY" \
  --checksum-authority "$CHECKSUM_AUTHORITY" \
  --output "$ARTIFACT_AUDIT"
```

S3 Batch ComputeChecksum is used because the source, non-VQ package, and
accepted routed baseline are hundreds of gigabytes. The completion report—not
`HeadObject`—is the full-object SHA-256 authority. The audit also range-reads
every safetensors header and proves that each declared tensor payload is
non-empty, contiguous, shape/dtype-consistent, and inside the exact object
size.

If a pinned source shard is absent, use
`restore_missing_hf_shards_to_s3.py`. It streams the exact Hugging Face
revision into an unpublished multipart upload, resumes truncated HTTP streams
by byte range, checks the pinned LFS byte count and SHA-256, and only then
completes the S3 object. It never overwrites an unauthenticated existing
object.

The first paid job is `submit_sky_campaign.sh --cache-seed`. It is permitted
only with the explicit all-zero cache placeholder and produces one real,
content-addressed schema-v3 qualification cache. Production and H100
qualification refuse that placeholder. After the seed marker authenticates,
build and stage a new descriptor bound to the real cache prefix and manifest
SHA-256. Cache objects are uploaded with full-object SHA-256 and
`If-None-Match: *`; the ready marker is published last.
Set `REPO_TAR_SOURCE` to the seed bundle's `repo.tar.gz` when building that
post-seed bundle so qualification and production consume the exact code that
created the cache rather than a newly packed dirty worktree.

When building either bundle, use the identities with the semantics the worker
validates: `SOURCE_SNAPSHOT_SHA256` is the pinned source
`model.safetensors.index.json` SHA-256; `NON_VQ_PACKAGE_SHA256` is
`package_set_sha256` from `non-vq-manifest.json`; and
`TRAINING_BASELINE_SHA256` is the exact accepted
`training-baseline.json` file SHA-256. The production inventory separately
binds every manifest, shard, and safetensors payload.

SkyPilot does not consume the legacy launch-template user data. Its bootstrap
therefore runs `skypilot/prepare_nvme_storage.sh` before downloading model
artifacts. The preflight reuses the DLAMI-managed instance-store volume when
present, safely prepares only unclaimed EC2 instance-store devices as a
fallback, and refuses to continue if `/mnt/nvme` resolves to the 300 GiB root
filesystem or has less than 700 GiB free.

## What the stack creates

| Resource | Purpose |
|---|---|
| VPC + 1 public subnet (1 AZ) | Isolated network for the GPU node |
| **S3 gateway endpoint** | Free, multi-GB/s S3↔EC2 transfer (no NAT, no egress $) |
| S3 bucket `keep-glm52-models-<acct>-<region>` | Model + artifact staging (**retained** on teardown; historical parameterization only—alternate regions are forbidden for the current campaign) |
| IAM role + instance profile | SSM access + read/write to the model bucket only |
| Security group | **Egress only** — access is via SSM Session Manager, no SSH keys |
| Launch template | **HISTORICAL / FORBIDDEN FOR CURRENT CAMPAIGN:** raw launch template with NVMe RAID0, s5cmd, and Spot toggle |

## Historical GPU options & pricing — forbidden for the current campaign
> **FORBIDDEN FOR THE CURRENT CAMPAIGN:** The table below preserves historical
> pricing research; it is not a worker-shape, lifecycle, or region menu. The
> only approved worker is one on-demand `p5.48xlarge` in `us-west-2`, provisioned
> by the guarded SkyPilot submission entrypoint.


| Instance | GPUs | On-demand | Spot | Historical use |
|---|---|---|---|---|
| `g6e.xlarge` | 1× L40S 48GB | $1.86/hr | ~$0.90 | **FORBIDDEN FOR CURRENT CAMPAIGN** — historical spike / seeder |
| `p5.48xlarge` | 8× H100 640GB | $55.04/hr | ~$15.9/hr | On-demand is the approved shape only through guarded SkyPilot; **the listed Spot path is forbidden** |
| `p4de.24xlarge` | 8× A100 640GB | $27.45/hr | ~$18.5/hr | **FORBIDDEN FOR CURRENT CAMPAIGN** — alternate worker shape |
| `g6e.48xlarge` | 8× L40S 384GB | ~$30/hr | ~$8–14/hr | **FORBIDDEN FOR CURRENT CAMPAIGN** — alternate worker shape and Spot |

The historical quota and cost estimate was: On-Demand P/G = 768 vCPU, with an
estimated full teacher-gen run of **2–6 hr → ~$110–330 on-demand, ~$32–96
Spot**. **The Spot estimate and any attempt to route around the guarded
SkyPilot submission are forbidden for the current campaign.**

## Runbook

This is the legacy raw-EC2 runbook and is **forbidden for the current
campaign**. Use the guarded
`aws/glm52-gpu/scripts/submit_sky_campaign.sh --production` entrypoint instead.
The historical prerequisites were the `aws` CLI logged in through the explicit
`keep-gpu` profile and the SSM Session Manager plugin.

```bash
cd aws/glm52-gpu
export AWS_PROFILE=keep-gpu
./scripts/assert_rnd_aws_account.sh

# 1. Deploy the stack ($0)
./scripts/deploy.sh

# 2. Upload the small KEEP-local artifacts from your Mac (non-VQ pkg, profile, teich pack)
./scripts/stage_local_artifacts.sh

# 3. HISTORICAL / FORBIDDEN FOR CURRENT CAMPAIGN: seed on a g6e Spot box.
#    The g6e worker shape, Spot lifecycle, and raw launch path are all forbidden.
INSTANCE_TYPE=g6e.xlarge USE_SPOT=1 ./scripts/launch_instance.sh  # FORBIDDEN FOR CURRENT CAMPAIGN
ID=<printed-id> ./scripts/connect.sh
#   on the box:
BUCKET=keep-glm52-models-246813579024-us-west-2 ./seed_from_hf.sh   # (copy script over or git clone)

# 4. HISTORICAL / FORBIDDEN FOR CURRENT CAMPAIGN: raw-launch the H100 box.
INSTANCE_TYPE=p5.48xlarge ./scripts/launch_instance.sh  # FORBIDDEN; use guarded SkyPilot submission
ID=<printed-id> ./scripts/connect.sh
#   on the box:
BUCKET=keep-glm52-models-246813579024-us-west-2 ./pull_from_s3.sh   # ~250GB in 1-3 min
#   ... then run the teacher-gen driver (task #10) against /mnt/nvme/*

# 5. HISTORICAL / FORBIDDEN FOR CURRENT CAMPAIGN: raw EC2 termination path.
aws ec2 terminate-instances --region us-west-2 --instance-ids <id>  # FORBIDDEN; use guarded SkyPilot break glass

# 6. Optional: tear down the stack (S3 bucket is retained)
./scripts/teardown.sh
```

## S3 layout

```
s3://keep-glm52-models-246813579024-us-west-2/
  source-snapshot/    # REAP-504B weights + config + index + tokenizer (from HF)
  non-vq-package/      # KEEP-derived non-expert tensors (from Mac)
  profile/             # glm52-reap-504b-v2.yaml (from Mac)
  teich-pack/          # coding-agent prompt pack (from Mac)
  teacher-cache-out/   # results written back here
```

## Current execution boundary

The resumable teacher generator, schema-v3 sparse cache finalizer, adapter
trainer, spend ledger, watchdog, and SkyPilot Managed Job entrypoint are wired.
No production worker may start from the dirty worktree: build a content-addressed
repository tar, stage the descriptor last, pass the S3 full-object checksum
audit, and rehearse those exact staged objects first.

The first paid step is the bounded one-row qualification-cache seed. It may
consume at most six cumulative GPU hours from the same 24-hour authorization.
H100 replacement-node qualification and production use the exact same repository
tar and may proceed only after the seed marker is authenticated and a new
descriptor is bound to that real cache.
