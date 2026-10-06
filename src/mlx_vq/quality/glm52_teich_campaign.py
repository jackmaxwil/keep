"""Compatibility alias for the enforcement-native campaign state."""

import sys
from glm52_enforcement import glm52_teich_campaign as _native

sys.modules[__name__] = _native
