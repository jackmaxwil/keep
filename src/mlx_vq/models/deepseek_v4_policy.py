"""Compatibility alias: the canonical module is ``ramp.models.deepseek_v4_policy``.

Not a re-export -- ``sys.modules`` is rebound so both names refer to the same
module object, and therefore to the same precision policy.
"""

import sys

from keep._alias import alias_module

sys.modules[__name__] = alias_module(__name__, "ramp.models.deepseek_v4_policy")
