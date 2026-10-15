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


def cleanup_shared_renderer() -> None:
    """Close the hidden Open3D window and release the shared renderer."""
    global _renderer, _HIDDEN_WINDOW
    if _HIDDEN_WINDOW is not None:
        try:
            _HIDDEN_WINDOW.close()
        except Exception:
            pass
        _HIDDEN_WINDOW = None
    _renderer = None
    logger.debug("Shared renderer cleaned up")


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


_shared_offscreen_renderer: Optional[object] = None
"""Cached OffscreenRenderer singleton for headless 3D point cloud rendering."""


def get_shared_offscreen_renderer(width: int = 800, height: int = 600):
    """
    Return the process-wide shared Open3D OffscreenRenderer singleton.

    Open3D's Filament rendering engine on Linux (EGL headless) only supports
    a single OffscreenRenderer per process. Creating multiple instances or
    destroying and recreating an OffscreenRenderer corrupts the EGL driver
    context and causes fatal segmentation faults (FEngine::loop).

    All 3D point cloud views (View3D, PreviewDialog, FilterDialog) must share
    this singleton and clear geometries between uses without deleting the renderer.
    """
    global _shared_offscreen_renderer
    if _shared_offscreen_renderer is not None:
        return _shared_offscreen_renderer

    import os
    if "__EGL_VENDOR_LIBRARY_DIRS" not in os.environ and "EGL_VENDOR_LIBRARY_DIRS" not in os.environ:
        for _egl_dir in ("/usr/lib/x86_64-linux-gnu/GL/glvnd/egl_vendor.d", "/usr/share/glvnd/egl_vendor.d"):
            if os.path.isdir(_egl_dir):
                os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = _egl_dir
                break

    try:
        import open3d.visualization.rendering as o3d_render
    except ImportError:
        logger.warning("Open3D rendering not available")
        return None

    try:
        _shared_offscreen_renderer = o3d_render.OffscreenRenderer(width, height)
        if hasattr(_shared_offscreen_renderer, "scene") and _shared_offscreen_renderer.scene is not None:
            _shared_offscreen_renderer.scene.set_background([0.10, 0.10, 0.18, 1.0])
        logger.info("Created shared OffscreenRenderer singleton (%dx%d)", width, height)
    except Exception as exc:
        logger.warning("Failed to create shared OffscreenRenderer: %s", exc)
        _shared_offscreen_renderer = None

    return _shared_offscreen_renderer
