"""Compatibility layer for the historical rasterizer module name.

New code should import :mod:`socgs.rgb_ms.rasterization`.
"""

from .rasterization import *  # noqa: F401,F403
