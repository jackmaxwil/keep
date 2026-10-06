"""Text-level contract tests for the GPU teacher CloudFormation template.

The template uses CFN YAML short tags (!Ref, !GetAtt) that plain yaml.safe_load
rejects, so assertions are textual.
"""

import re
from pathlib import Path

TEMPLATE = Path("aws/glm52-gpu/cfn/gpu-teacher-stack.yaml").read_text()


def test_watchdog_rule_state_is_parameterized():
    assert "SkyWatchdogRuleState:" in TEMPLATE
    parameter_block = TEMPLATE.split("SkyWatchdogRuleState:", 1)[1][:400]
    assert "Default: DISABLED" in parameter_block
    assert "ENABLED" in parameter_block  # allowed value


def test_watchdog_rule_references_the_parameter():
    match = re.search(
        r"SkyWatchdogRule:\n(?:.*\n)*?      State: (.+)", TEMPLATE
    )
    assert match is not None, "SkyWatchdogRule resource not found"
    assert match.group(1).strip() == "!Ref SkyWatchdogRuleState"
