#!/usr/bin/env bash
# Refuse launch until Jack's campaign-alert subscription has a confirmed ARN.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TOPIC_ARN="${CAMPAIGN_ALERT_TOPIC_ARN:?set CAMPAIGN_ALERT_TOPIC_ARN}"
ENDPOINT="${CAMPAIGN_ALERT_EMAIL:-operator@example.com}"
"$SCRIPT_DIR/assert_rnd_aws_account.sh" >/dev/null
RESULT=$(mktemp)
trap 'rm -f "$RESULT"' EXIT
aws sns list-subscriptions-by-topic \
  --topic-arn "$TOPIC_ARN" \
  --region us-west-2 \
  --profile "$AWS_PROFILE" \
  --output json > "$RESULT"
python3 - "$RESULT" "$TOPIC_ARN" "$ENDPOINT" <<'PY'
import json, sys
path, topic, endpoint = sys.argv[1:]
value = json.load(open(path))
matches = [
    item
    for item in value.get("Subscriptions", [])
    if item.get("TopicArn") == topic
    and item.get("Protocol") == "email"
    and item.get("Endpoint") == endpoint
]
if len(matches) != 1:
    raise SystemExit(f"expected one SNS email subscription for {endpoint}")
arn = matches[0].get("SubscriptionArn")
if not isinstance(arn, str) or not arn.startswith("arn:") or (
    arn == "PendingConfirmation"
):
    raise SystemExit(f"SNS email subscription for {endpoint} is not confirmed")
print(arn)
PY
