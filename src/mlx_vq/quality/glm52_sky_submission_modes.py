"""Compatibility alias for the enforcement-native submission-mode contract."""

import sys
from glm52_enforcement import glm52_sky_submission_modes as _native

sys.modules[__name__] = _native
