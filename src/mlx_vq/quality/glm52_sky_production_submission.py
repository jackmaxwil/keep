"""Compatibility alias for the enforcement-native submission contract."""

import sys
from glm52_enforcement import glm52_sky_production_submission as _native

sys.modules[__name__] = _native
