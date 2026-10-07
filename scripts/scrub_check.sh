#!/usr/bin/env bash
# Pre-push scrub gate. Fails if the tracked tree carries private network
# identifiers, credentials, or AWS account and organization IDs outside the
# documented placeholders, or any pattern from a machine-local denylist.
#
# The denylist (employer names, internal domains, original account IDs) lives
# outside the repo so the patterns themselves are never published:
#   ${KEEP_SCRUB_DENYLIST:-$HOME/.config/keep/scrub-denylist.txt}
# One PCRE per line, '#' starts a comment.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

fail=0
report() {
  if [ -n "$2" ]; then
    echo "SCRUB FAIL [$1]:"
    echo "$2" | cut -c1-200 | head -20
    fail=1
  fi
}
hits() { git grep -P -I -n -e "$1" -- . ':!scripts/scrub_check.sh' || true; }

# RFC 1918 hosts. 10.20.0.0/16 is the retired GPU stack VPC plan, and other 10.x
# values are allowed only as synthetic AWS VPC fixtures under aws/ and tests/.
report "private-ipv4" "$(hits '(?<![\d.])(192\.168|172\.(1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?!\d)')"
report "private-ipv4-10" "$(hits '(?<![\d.])10\.(?!20\.)\d{1,3}\.\d{1,3}\.\d{1,3}(?!\d)' | grep -v -E '^(aws|tests)/' || true)"
# MAC addresses, including the zero-stripped form macOS arp prints. The
# RFC 7042 documentation block 00:00:5e:00:53:xx is the only allowed value.
report "mac-address" "$(hits '(?<![0-9A-Fa-f:])([0-9A-Fa-f]{1,2}:){5}[0-9A-Fa-f]{1,2}(?![0-9A-Fa-f:])' |
  grep -v -i -E '(^|[^0-9a-f:])0{0,2}0?:0{0,2}0?:5e:0?0?:53:[0-9a-f]{1,2}' || true)"
report "aws-access-key" "$(hits 'AKIA[0-9A-Z]{16}' | grep -v 'AKIAABCDEFGHIJKLMNOP' || true)"
report "private-key" "$(hits '-----BEGIN [A-Z ]*PRIVATE KEY-----')"
# Account IDs in ARNs or account fields, and organization IDs, must be one of
# the placeholders the scrubbed history already uses.
allowed='246813579024|135792468013|111122223333|444455556666|123456789012|000000000000'
report "aws-account-id" "$(hits "(arn:aws[a-z-]*:[a-z0-9-]*:[a-z0-9-]*:|[Aa]ccount_?[Ii][Dd]\W{1,4})(?!(${allowed}|(\d)\3{11})\b)\d{12}\b")"
report "aws-org-id" "$(hits '\bo-(?!aa111bb222\b|example1234\b)[a-z0-9]{10,32}\b')"

denylist="${KEEP_SCRUB_DENYLIST:-$HOME/.config/keep/scrub-denylist.txt}"
if [ -f "$denylist" ] && grep -q -v -E '^[[:space:]]*(#|$)' "$denylist"; then
  while IFS= read -r pattern; do
    case "$pattern" in ''|'#'*) continue ;; esac
    report "denylist" "$(git grep -P -I -n -i -e "$pattern" -- . || true)"
  done < "$denylist"
else
  echo "warning: no denylist patterns in $denylist; employer patterns were not checked" >&2
fi

[ "$fail" -eq 0 ] && echo "scrub check: clean"
exit "$fail"
