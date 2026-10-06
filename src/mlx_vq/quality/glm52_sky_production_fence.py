"""Compatibility alias for enforcement-native fence authority."""

import sys
from glm52_enforcement import glm52_sky_production_fence as _native

sys.modules[__name__] = _native
