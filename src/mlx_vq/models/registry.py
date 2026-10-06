"""Compatibility alias: the canonical module is ``ramp.models.registry``.

Not a re-export -- ``sys.modules`` is rebound so both names refer to the same
module object, and therefore to the same family registry.
"""

import sys

from keep._alias import alias_module

sys.modules[__name__] = alias_module(__name__, "ramp.models.registry")
