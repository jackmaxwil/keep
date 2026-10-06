"""Compatibility alias for enforcement-native generation authority."""

import sys
from glm52_enforcement import glm52_sky_production_generation as _native

sys.modules[__name__] = _native
