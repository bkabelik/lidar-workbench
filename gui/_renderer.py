"""
Open3D Renderer provider — internal helper.

Open3D ≥0.19 removed ``SceneWidget.window``; a ``Renderer`` must now be
obtained from an ``gui.Window``.  This module creates a single hidden
window on first use and shares its ``Renderer``, so that standalone
``SceneWidget`` instances (embedded into Qt via ``createWindowContainer``)
can construct their ``Open3DScene`` objects.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("lidar_workbench.gui._renderer")

_renderer: Optional[object] = None
"""Cached Renderer from a hidden Open3D Window."""

_HIDDEN_WINDOW: Optional[object] = None
"""Hidden Open3D Window that owns the Renderer."""


def get_shared_renderer():
    """
    Return a :class:`open3d.visualization.rendering.Renderer` that can
    be used to construct ``Open3DScene`` objects for embedded
    ``SceneWidget`` instances.

    A single hidden Open3D window is created on the first call and
    kept alive for the lifetime of the process.
    """
    global _renderer, _HIDDEN_WINDOW

    if _renderer is not None:
        return _renderer

    try:
        import open3d.visualization.gui as o3d_gui
    except ImportError:
        raise RuntimeError("Open3D is not available")

    # Create a tiny hidden window just to obtain its Renderer
    try:
        win = o3d_gui.Application.instance.create_window(
            "_hidden_renderer", 1, 1
        )
        win.show(False)  # hide immediately
        _renderer = win.renderer
        _HIDDEN_WINDOW = win
        logger.debug("Created hidden Open3D window for shared Renderer")
    except Exception as exc:
        logger.error("Failed to create hidden Open3D window: %s", exc)
        raise RuntimeError(
            "Cannot obtain Open3D Renderer — Open3D GUI initialisation may have failed"
        ) from exc

    return _renderer
