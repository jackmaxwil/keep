from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "aws/glm52-gpu/scripts/sky_campaign_break_glass.sh"
RUNBOOK = ROOT / "aws/glm52-gpu/SKYPILOT_BREAK_GLASS.md"


def test_sky_break_glass_is_account_guarded_bounded_and_has_no_raw_launch() -> None:
    source = SCRIPT.read_text()
    assert "assert_rnd_aws_account.sh" in source
    assert "validate_sky_campaign_descriptor" in source
    assert "sky jobs queue" in source
    assert "sky jobs logs" in source
    assert "systemctl" in source
    assert "journalctl" in source
    assert "touch /run/keep-glm52/STOP" in source
    assert "sky jobs cancel" in source
    assert "--graceful-timeout 900" in source
    assert "validate_sky_terminal_state" in source
    assert "GPU_SPEND_LEDGER.jsonl" in source
    assert "CAMPAIGN_DRAINED.json" in source
    assert "describe-volumes" in source
    assert "describe-addresses" in source
    assert "RunInstances" not in source
    assert "run-instances" not in source
    assert "PurchaseCapacityBlock" not in source
    assert "request-spot" not in source


def test_sky_break_glass_runbook_covers_exact_operator_recovery_path() -> None:
    source = RUNBOOK.read_text()
    for required in (
        "246813579024",
        "sky jobs queue",
        "controller logs",
        "SSM",
        "graceful-stop",
        "cancel",
        "CAMPAIGN_DRAINED.json",
        "GPU_SPEND_LEDGER.jsonl",
        "same run ID",
        "remaining approved balance",
        "orphan",
        "Capacity Block",
    ):
        assert required in source
