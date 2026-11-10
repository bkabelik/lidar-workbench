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
    global _renderer, _HIDDEN_WINDOW, _shared_offscreen_renderer
    if _HIDDEN_WINDOW is not None:
        try:
            _HIDDEN_WINDOW.close()
        except Exception:
            pass
        _HIDDEN_WINDOW = None
    _renderer = None
    if _shared_offscreen_renderer is not None:
        if hasattr(_shared_offscreen_renderer, "close"):
            try:
                _shared_offscreen_renderer.close()
            except Exception:
                pass
        _shared_offscreen_renderer = None
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


class _VisualizerSceneAdapter:
    """Scene compatibility adapter wrapping Open3D Visualizer."""

    def __init__(self, vis):
        self._vis = vis
        self._geometries = {}

    def add_geometry(self, name: str, geom, mat=None):
        import open3d as o3d
        if name in self._geometries:
            try:
                self._vis.remove_geometry(self._geometries[name], reset_bounding_box=False)
            except Exception:
                pass
        self._geometries[name] = geom
        first = len(self._geometries) == 1
        try:
            self._vis.add_geometry(geom, reset_bounding_box=first)
        except Exception:
            try:
                self._vis.update_geometry(geom)
            except Exception:
                pass

    def remove_geometry(self, name: str):
        if name in self._geometries:
            geom = self._geometries.pop(name)
            try:
                self._vis.remove_geometry(geom, reset_bounding_box=False)
            except Exception:
                pass

    def clear_geometry(self):
        for geom in list(self._geometries.values()):
            try:
                self._vis.remove_geometry(geom, reset_bounding_box=False)
            except Exception:
                pass
        self._geometries.clear()

    def has_geometry(self, name: str) -> bool:
        return name in self._geometries

    def set_background(self, color):
        import numpy as np
        try:
            opt = self._vis.get_render_option()
            if opt is not None and len(color) >= 3:
                opt.background_color = np.array(color[:3], dtype=np.float64)
        except Exception:
            pass


class VisualizerOffscreenAdapter:
    """
    OpenGL-based offscreen renderer adapter for Windows where Filament's
    Vulkan headless swapchain readback yields blank (all-zero) frames.
    """

    def __init__(self, width: int = 800, height: int = 600):
        import open3d as o3d
        import numpy as np
        self._width = width
        self._height = height
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Offscreen3D", visible=False, width=width, height=height)
        opt = self._vis.get_render_option()
        if opt is not None:
            opt.background_color = np.array([0.10, 0.10, 0.18], dtype=np.float64)
            opt.point_size = 2.5
        self.scene = _VisualizerSceneAdapter(self._vis)

    def setup_camera(self, fov: float, center, eye, up, near: float = -1.0, far: float = -1.0):
        import numpy as np
        try:
            ctr = self._vis.get_view_control()
            if ctr is None:
                return
            params = ctr.convert_to_pinhole_camera_parameters()
            c = np.asarray(center, dtype=np.float64)
            e = np.asarray(eye, dtype=np.float64)
            u = np.asarray(up, dtype=np.float64)

            front = c - e
            norm = np.linalg.norm(front)
            if norm > 1e-9:
                front = front / norm
            else:
                front = np.array([0.0, 1.0, 0.0])

            norm_u = np.linalg.norm(u)
            up_vec = u / (norm_u if norm_u > 1e-9 else 1.0)

            right = np.cross(front, up_vec)
            norm_r = np.linalg.norm(right)
            right = right / (norm_r if norm_r > 1e-9 else 1.0)

            true_up = np.cross(right, front)

            # Open3D camera coordinates: X right, Y down, Z forward
            R = np.vstack([right, -true_up, front])
            t = -R @ e

            extrinsic = np.eye(4, dtype=np.float64)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t

            params.extrinsic = extrinsic
            ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        except Exception as exc:
            logger.debug("Visualizer setup_camera warning: %s", exc)

    def render_to_image(self):
        import open3d as o3d
        import numpy as np
        try:
            self._vis.poll_events()
            self._vis.update_renderer()
            img = self._vis.capture_screen_float_buffer(do_render=True)
            arr = (np.asarray(img) * 255).astype(np.uint8)
            return o3d.geometry.Image(arr)
        except Exception as exc:
            logger.warning("Visualizer render_to_image failed: %s", exc)
            return o3d.geometry.Image(np.zeros((self._height, self._width, 3), dtype=np.uint8))

    def close(self):
        try:
            self._vis.destroy_window()
        except Exception:
            pass


def get_shared_offscreen_renderer(width: int = 800, height: int = 600):
    """
    Return the process-wide shared Open3D OffscreenRenderer singleton.

    On Linux, uses Filament OffscreenRenderer (with shared singleton semantics
    to avoid FEngine::loop context crashes).
    On Windows, uses VisualizerOffscreenAdapter (OpenGL FBO) to bypass Filament
    Vulkan headless swapchain readback limitations.
    """
    global _shared_offscreen_renderer
    if _shared_offscreen_renderer is not None:
        return _shared_offscreen_renderer

    import sys
    import os

    # On Windows, Filament Vulkan headless render_to_image returns black frames.
    # Use OpenGL Visualizer adapter directly.
    if sys.platform == "win32":
        try:
            _shared_offscreen_renderer = VisualizerOffscreenAdapter(width, height)
            logger.info("Created Windows OpenGL offscreen renderer adapter (%dx%d)", width, height)
            return _shared_offscreen_renderer
        except Exception as exc:
            logger.warning("Failed to create VisualizerOffscreenAdapter: %s, falling back to Filament", exc)

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
        try:
            _shared_offscreen_renderer = VisualizerOffscreenAdapter(width, height)
            logger.info("Fell back to VisualizerOffscreenAdapter (%dx%d)", width, height)
        except Exception:
            _shared_offscreen_renderer = None

    return _shared_offscreen_renderer
