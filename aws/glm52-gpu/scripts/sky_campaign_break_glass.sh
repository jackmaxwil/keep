#!/usr/bin/env bash
# Guarded operator controls for one authenticated SkyPilot campaign.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
ACTION="${1:-}"
DESCRIPTOR="${CAMPAIGN_DESCRIPTOR:?set CAMPAIGN_DESCRIPTOR}"
SKY_BIN="${SKY_BIN:?set SKY_BIN to SkyPilot 0.13.0}"
SKY_CONFIG="${SKYPILOT_CONFIG:?set SKYPILOT_CONFIG to the rendered config}"
REGION=us-west-2

usage() {
  echo "usage: $0 {status|logs|inspect-worker|graceful-stop|cancel|verify|audit-orphans|resume}" >&2
  exit 64
}
case "$ACTION" in
  status|logs|inspect-worker|graceful-stop|cancel|verify|audit-orphans|resume) ;;
  *) usage ;;
esac

"$SCRIPT_DIR/assert_rnd_aws_account.sh" >/dev/null
read -r RUN_ID BUCKET <<EOF
$(python3 - "$DESCRIPTOR" "$REPO" <<'PY'
import importlib.util, json, pathlib, sys
descriptor_path = pathlib.Path(sys.argv[1])
module_path = pathlib.Path(sys.argv[2]) / "src/mlx_vq/quality/glm52_sky_campaign.py"
spec = importlib.util.spec_from_file_location("_glm52_break_glass", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
value = module.validate_sky_campaign_descriptor(
    json.loads(descriptor_path.read_bytes())
)
print(value["run_id"], value["bucket"])
PY
)
EOF

sky() {
  AWS_PROFILE="$AWS_PROFILE" "$SKY_BIN" "$@" --config "$SKY_CONFIG"
}

worker_id() {
  local value
  value=$(aws ec2 describe-instances \
    --profile "$AWS_PROFILE" --region "$REGION" \
    --filters \
      "Name=tag:campaign-run-id,Values=$RUN_ID" \
      "Name=instance-type,Values=p5.48xlarge" \
      "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text)
  set -- $value
  if [ "$#" -ne 1 ]; then
    echo "expected exactly one active campaign worker; found $#" >&2
    return 2
  fi
  printf '%s\n' "$1"
}

ssm_commands() {
  local instance_id command_json command_id
  instance_id=$(worker_id)
  command_json="$1"
  command_id=$(aws ssm send-command \
    --profile "$AWS_PROFILE" --region "$REGION" \
    --instance-ids "$instance_id" \
    --document-name AWS-RunShellScript \
    --comment "KEEP GLM-5.2 Sky break glass: $ACTION" \
    --parameters "$command_json" \
    --query Command.CommandId --output text)
  aws ssm wait command-executed \
    --profile "$AWS_PROFILE" --region "$REGION" \
    --command-id "$command_id" --instance-id "$instance_id" || true
  aws ssm get-command-invocation \
    --profile "$AWS_PROFILE" --region "$REGION" \
    --command-id "$command_id" --instance-id "$instance_id" --output json
}

case "$ACTION" in
  status)
    sky jobs queue --output json
    aws ec2 describe-instances \
      --profile "$AWS_PROFILE" --region "$REGION" \
      --filters "Name=tag:campaign-run-id,Values=$RUN_ID" \
      --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime]' \
      --output table
    ;;
  logs)
    sky jobs logs -n "$RUN_ID" --controller --no-follow --tail 300
    sky jobs logs -n "$RUN_ID" --no-follow --tail 300
    ;;
  inspect-worker)
    ssm_commands '{"commands":["systemctl status keep-glm52-campaign.service --no-pager || true","journalctl -u keep-glm52-campaign.service -n 300 --no-pager || true","cat /mnt/nvme/glm52-campaign/runtime/GPU_SPEND_STATUS.json 2>/dev/null || true","cat /mnt/nvme/glm52-campaign/monitor/heartbeat.json 2>/dev/null || true"]}'
    ;;
  graceful-stop)
    ssm_commands '{"commands":["sudo install -d -m 0755 /run/keep-glm52","sudo touch /run/keep-glm52/STOP","sudo systemctl kill --signal=TERM keep-glm52-campaign.service || true"]}'
    ;;
  cancel)
    "$0" graceful-stop
    sky jobs cancel -n "$RUN_ID" --graceful --graceful-timeout 900 -y
    ;;
  verify)
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT
    aws s3 cp "s3://$BUCKET/campaigns/$RUN_ID/CAMPAIGN_DRAINED.json" \
      "$work/CAMPAIGN_DRAINED.json" --profile "$AWS_PROFILE" \
      --region "$REGION" --only-show-errors
    aws s3 cp "s3://$BUCKET/campaigns/$RUN_ID/ledger/campaign-ledger.jsonl" \
      "$work/campaign-ledger.jsonl" --profile "$AWS_PROFILE" \
      --region "$REGION" --only-show-errors
    aws s3 cp "s3://$BUCKET/campaigns/$RUN_ID/runtime/GPU_SPEND_LEDGER.jsonl" \
      "$work/GPU_SPEND_LEDGER.jsonl" --profile "$AWS_PROFILE" \
      --region "$REGION" --only-show-errors
    python3 - "$RUN_ID" "$work" "$REPO" <<'PY'
import importlib.util, json, pathlib, sys
run_id, root_raw, repo_raw = sys.argv[1:]
root = pathlib.Path(root_raw); repo = pathlib.Path(repo_raw)
def load(name, relative):
    path = repo / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
sky = load("_break_sky", "src/mlx_vq/quality/glm52_sky_campaign.py")
terminal = load("_break_terminal", "src/mlx_vq/quality/glm52_sky_terminal_state.py")
spend_raw = (root / "GPU_SPEND_LEDGER.jsonl").read_bytes()
first = json.loads(spend_raw.splitlines()[0])
ledger = sky.GpuSpendLedger(
    root / "GPU_SPEND_LEDGER.jsonl",
    run_id=run_id,
    approval_sha256=first["approval_sha256"],
    approved_gpu_runtime_seconds=sky.APPROVED_GPU_RUNTIME_SECONDS,
    approved_gpu_cost_usd=sky.APPROVED_GPU_COST_USD,
    hourly_cost_usd=sky.APPROVED_HOURLY_COST_USD,
)
drained_raw = (root / "CAMPAIGN_DRAINED.json").read_bytes()
drained = json.loads(drained_raw)
report = terminal.validate_sky_terminal_state(
    run_id=run_id,
    execution_deadline=drained["execution_deadline"],
    gpu_spend_authority_sha256=ledger.genesis_sha256,
    drained_raw=drained_raw,
    campaign_ledger_raw=(root / "campaign-ledger.jsonl").read_bytes(),
    spend_ledger_raw=spend_raw,
)
print(json.dumps(report, sort_keys=True, indent=2))
PY
    ;;
  audit-orphans)
    aws ec2 describe-instances \
      --profile "$AWS_PROFILE" --region "$REGION" \
      --filters "Name=tag:campaign-run-id,Values=$RUN_ID" \
      --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' \
      --output table
    aws ec2 describe-volumes \
      --profile "$AWS_PROFILE" --region "$REGION" \
      --filters "Name=tag:campaign-run-id,Values=$RUN_ID" \
      --query 'Volumes[].[VolumeId,State,Size,Attachments[0].InstanceId]' \
      --output table
    aws ec2 describe-addresses \
      --profile "$AWS_PROFILE" --region "$REGION" \
      --filters "Name=tag:campaign-run-id,Values=$RUN_ID" \
      --query 'Addresses[].[AllocationId,PublicIp,InstanceId]' --output table
    ;;
  resume)
    exec "$SCRIPT_DIR/submit_sky_campaign.sh"
    ;;
esac
