"""
LiDAR Workbench — Manual Editing Module.

Implements profile extraction, point selection tools, class assignment,
and undo/redo command stack — modelled after Terrasolid TerraScan's
manual editing workflow.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import DEFAULT_PROFILE_WIDTH_M
from .tile_manager import TileManager

logger = logging.getLogger("lidar_workbench.manual_edit")


# ── Command pattern for undo/redo ──────────────────────────────────────


class EditCommand(ABC):
    """Abstract base for an undoable classification edit."""

    @abstractmethod
    def execute(self) -> bool:
        """Perform the edit. Returns ``True`` on success."""
        ...

    @abstractmethod
    def undo(self) -> bool:
        """Reverse the edit. Returns ``True`` on success."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the command."""
        ...


@dataclass
class ClassifyPointsCommand(EditCommand):
    """
    Reclassify a set of points.

    Stores the old and new class values so the operation can be
    undone by swapping them back.
    """

    tile_manager: TileManager
    tile_id: str
    point_indices: np.ndarray
    old_classes: np.ndarray
    new_class: int

    def execute(self) -> bool:
        return self.tile_manager.update_tile_classifications(
            self.tile_id, self.point_indices, self.new_class
        )

    def undo(self) -> bool:
        # Restore original classes
        result = True
        for code in np.unique(self.old_classes):
            mask = self.old_classes == code
            indices_subset = self.point_indices[mask]
            if len(indices_subset) == 0:
                continue
            ok = self.tile_manager.update_tile_classifications(
                self.tile_id, indices_subset, int(code)
            )
            result = result and ok
        return result

    @property
    def description(self) -> str:
        return (
            f"Classify {len(self.point_indices)} point(s) "
            f"→ class {self.new_class}"
        )


class EditStack:
    """
    Bounded undo/redo stack for classification commands.

    Maintains two stacks: *undo_stack* (commands that can be undone)
    and *redo_stack* (commands that were undone and can be redone).
    """

    def __init__(self, max_size: int = 100) -> None:
        self._undo: deque[EditCommand] = deque(maxlen=max_size)
        self._redo: deque[EditCommand] = deque(maxlen=max_size)

    def push(self, command: EditCommand) -> None:
        """Execute *command* and push it onto the undo stack."""
        if command.execute():
            self._undo.append(command)
            self._redo.clear()  # new action invalidates redo history
            logger.debug("Undo stack: %d, redo cleared", len(self._undo))
        else:
            logger.warning("Command execution failed — not pushed to undo stack")

    def undo(self) -> Optional[EditCommand]:
        """Undo the most recent command. Returns it, or ``None``."""
        if not self._undo:
            return None
        cmd = self._undo.pop()
        if cmd.undo():
            self._redo.append(cmd)
            logger.debug("Undo: %s", cmd.description)
            return cmd
        else:
            logger.warning("Undo failed for: %s", cmd.description)
            self._undo.append(cmd)  # put it back
            return None

    def redo(self) -> Optional[EditCommand]:
        """Redo the most recently undone command."""
        if not self._redo:
            return None
        cmd = self._redo.pop()
        if cmd.execute():
            self._undo.append(cmd)
            logger.debug("Redo: %s", cmd.description)
            return cmd
        else:
            self._redo.append(cmd)
            return None

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo) > 0

    @property
    def undo_stack_size(self) -> int:
        return len(self._undo)

    @property
    def redo_stack_size(self) -> int:
        return len(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()


# ── Profile extraction ─────────────────────────────────────────────────


@dataclass
class ProfileData:
    """
    Points extracted along a profile line.

    Attributes:
        distances:   Distance along the profile for each point (meters).
        elevations:  Point elevation (Z).
        xs:          Original X coordinates.
        ys:          Original Y coordinates.
        classifications: ASPRS class codes.
        intensities:     Intensity values.
        indices:     Indices into the original tile point arrays.
    """

    distances: np.ndarray
    elevations: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    classifications: np.ndarray
    intensities: np.ndarray
    indices: np.ndarray


def extract_profile(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    classifications: np.ndarray,
    intensities: np.ndarray,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    width: float = DEFAULT_PROFILE_WIDTH_M,
) -> ProfileData:
    """
    Extract points that lie within *width* meters of a profile line.

    Points are projected onto the profile line to compute their *distance*
    along the profile, and their perpendicular offset is used to filter
    by the corridor width.

    Args:
        xs, ys, zs:           Point coordinate arrays.
        classifications:      ASPRS class codes.
        intensities:          Intensity values.
        start_xy, end_xy:     Profile endpoints in CRS units ``(x, y)``.
        width:                Half-width of the extraction corridor (meters).

    Returns:
        :class:`ProfileData` with the extracted points.
    """
    n = len(xs)
    if n == 0:
        return ProfileData(
            distances=np.array([], dtype=np.float64),
            elevations=np.array([], dtype=np.float64),
            xs=np.array([], dtype=np.float64),
            ys=np.array([], dtype=np.float64),
            classifications=np.array([], dtype=np.uint8),
            intensities=np.array([], dtype=np.uint16),
            indices=np.array([], dtype=np.int64),
        )

    sx, sy = start_xy
    ex, ey = end_xy

    # Direction vector of profile line
    dx = ex - sx
    dy = ey - sy
    line_len_sq = dx * dx + dy * dy

    if line_len_sq < 1e-9:
        # Degenerate line — treat as point selection within radius
        px = xs - sx
        py = ys - sy
        perp = np.sqrt(px * px + py * py)
        mask = perp <= width
        distances = np.zeros_like(perp)
    else:
        # Project each point onto the line segment
        px = xs - sx
        py = ys - sy
        t = (px * dx + py * dy) / line_len_sq
        t = np.clip(t, 0.0, 1.0)

        # Closest point on segment
        cx = sx + t * dx
        cy = sy + t * dy

        # Perpendicular distance
        perp_x = xs - cx
        perp_y = ys - cy
        perp = np.sqrt(perp_x * perp_x + perp_y * perp_y)

        mask = perp <= (width / 2.0)
        distances = t * np.sqrt(line_len_sq)

    indices = np.where(mask)[0]

    logger.info(
        "Profile: %d points extracted (width=%.1f m, line=%.1f m)",
        len(indices), width, np.sqrt(line_len_sq),
    )

    return ProfileData(
        distances=distances[mask],
        elevations=zs[mask],
        xs=xs[mask],
        ys=ys[mask],
        classifications=classifications[mask],
        intensities=intensities[mask],
        indices=indices,
    )


# ── Selection tools ────────────────────────────────────────────────────

# Type alias for selection functions
SelectionFunc = Callable[
    [np.ndarray, np.ndarray, Any],
    np.ndarray,
]  # (distances, elevations, params) → boolean mask


def select_above_line(
    distances: np.ndarray,
    elevations: np.ndarray,
    line: Tuple[Tuple[float, float], Tuple[float, float]],
) -> np.ndarray:
    """
    Select points whose elevation is ABOVE a user-drawn line in the
    profile view (distance–elevation plane).

    Args:
        distances:  Distance-along-profile array.
        elevations: Elevation array.
        line:       ``((d1, z1), (d2, z2))`` — the line endpoints in
                    profile coordinates.

    Returns:
        Boolean mask where ``True`` means "above the line".
    """
    (d1, z1), (d2, z2) = line
    if abs(d2 - d1) < 1e-9:
        # Vertical line: points to the right are "above"
        return distances >= d1

    # Line equation: z = m * d + b
    m = (z2 - z1) / (d2 - d1)
    b = z1 - m * d1

    line_z = m * distances + b
    return elevations > line_z


def select_below_line(
    distances: np.ndarray,
    elevations: np.ndarray,
    line: Tuple[Tuple[float, float], Tuple[float, float]],
) -> np.ndarray:
    """
    Select points whose elevation is BELOW a user-drawn line.
    """
    (d1, z1), (d2, z2) = line
    if abs(d2 - d1) < 1e-9:
        return distances < d1

    m = (z2 - z1) / (d2 - d1)
    b = z1 - m * d1
    line_z = m * distances + b
    return elevations < line_z


def select_rectangle(
    distances: np.ndarray,
    elevations: np.ndarray,
    rect: Tuple[float, float, float, float],
) -> np.ndarray:
    """
    Select points inside a rectangle in profile coordinates.

    Args:
        rect: ``(d_min, d_max, z_min, z_max)``.
    """
    d_min, d_max, z_min, z_max = rect
    return (
        (distances >= d_min)
        & (distances <= d_max)
        & (elevations >= z_min)
        & (elevations <= z_max)
    )


def select_brush(
    distances: np.ndarray,
    elevations: np.ndarray,
    center: Tuple[float, float],
    radius: float = 2.0,
) -> np.ndarray:
    """
    Freehand brush selection — all points within *radius* of *center*
    in the profile (distance–elevation) plane.

    Args:
        center: ``(d_center, z_center)``.
        radius: Brush radius in profile units (meters).
    """
    dc, zc = center
    d_dist = distances - dc
    d_elev = elevations - zc
    return np.sqrt(d_dist * d_dist + d_elev * d_elev) <= radius


# ── Manual Editor ──────────────────────────────────────────────────────


class ManualEditor:
    """
    High-level manual editing controller.

    Manages the edit stack, profile state, and selection for a single
    open tile.  The GUI views communicate with this object to perform
    classification edits and track undo/redo state.

    Usage::

        editor = ManualEditor(tile_manager)
        editor.open_tile("tile_001")

        # Define a profile
        profile = editor.extract_profile((x1, y1), (x2, y2), width=5.0)

        # Select points above a line
        mask = editor.select_above_line(((d1, z1), (d2, z2)))

        # Assign new class
        editor.assign_class(3)  # Low Vegetation

        # Undo
        editor.undo()
    """

    def __init__(self, tile_manager: TileManager) -> None:
        self._tm = tile_manager
        self._edit_stack = EditStack(max_size=200)

        # Current tile state
        self._tile_id: Optional[str] = None
        self._point_data: Optional[Dict[str, np.ndarray]] = None

        # Current profile
        self._profile: Optional[ProfileData] = None

        # Current selection (local indices into self._profile)
        self._selected_indices: Optional[np.ndarray] = None

    # ── tile management ────────────────────────────────────────────

    def open_tile(self, tile_id: str) -> bool:
        """
        Load a tile for editing.

        Returns ``True`` on success, ``False`` if the tile cannot be loaded.
        """
        data = self._tm.load_tile_points_full(tile_id)
        if data is None:
            logger.error("Failed to load tile %s", tile_id)
            return False

        self._tile_id = tile_id
        self._point_data = data
        self._profile = None
        self._selected_indices = None
        self._edit_stack.clear()
        logger.info("Opened tile %s for editing (%d points)", tile_id, len(data["x"]))
        return True

    @property
    def tile_id(self) -> Optional[str]:
        return self._tile_id

    @property
    def point_data(self) -> Optional[Dict[str, np.ndarray]]:
        return self._point_data

    # ── profile extraction ─────────────────────────────────────────

    def extract_profile(
        self,
        start_xy: Tuple[float, float],
        end_xy: Tuple[float, float],
        width: float = DEFAULT_PROFILE_WIDTH_M,
    ) -> Optional[ProfileData]:
        """
        Extract points along a profile line from the currently loaded tile.

        Returns ``None`` if no tile is loaded.
        """
        if self._point_data is None:
            logger.warning("No tile loaded — cannot extract profile")
            return None

        self._profile = extract_profile(
            self._point_data["x"],
            self._point_data["y"],
            self._point_data["z"],
            self._point_data["classification"],
            self._point_data["intensity"],
            start_xy,
            end_xy,
            width,
        )
        self._selected_indices = None
        return self._profile

    @property
    def profile(self) -> Optional[ProfileData]:
        return self._profile

    # ── selection ──────────────────────────────────────────────────

    def select_above_line(
        self,
        line: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> np.ndarray:
        """Select points above a line in the profile view."""
        if self._profile is None or len(self._profile.distances) == 0:
            return np.array([], dtype=bool)
        self._selected_indices = select_above_line(
            self._profile.distances, self._profile.elevations, line
        )
        return self._selected_indices

    def select_below_line(
        self,
        line: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> np.ndarray:
        """Select points below a line in the profile view."""
        if self._profile is None or len(self._profile.distances) == 0:
            return np.array([], dtype=bool)
        self._selected_indices = select_below_line(
            self._profile.distances, self._profile.elevations, line
        )
        return self._selected_indices

    def select_rectangle(
        self,
        d_min: float,
        d_max: float,
        z_min: float,
        z_max: float,
    ) -> np.ndarray:
        """Select points inside a rectangle in the profile view."""
        if self._profile is None or len(self._profile.distances) == 0:
            return np.array([], dtype=bool)
        self._selected_indices = select_rectangle(
            self._profile.distances,
            self._profile.elevations,
            (d_min, d_max, z_min, z_max),
        )
        return self._selected_indices

    def select_brush(
        self,
        center: Tuple[float, float],
        radius: float = 2.0,
    ) -> np.ndarray:
        """Select points within a brush radius in the profile view."""
        if self._profile is None or len(self._profile.distances) == 0:
            return np.array([], dtype=bool)
        self._selected_indices = select_brush(
            self._profile.distances,
            self._profile.elevations,
            center,
            radius,
        )
        return self._selected_indices

    def clear_selection(self) -> None:
        """Deselect all points."""
        self._selected_indices = None

    def set_selection(self, mask: np.ndarray) -> None:
        """
        Set the current selection from an external source (e.g. profile view).

        Args:
            mask: Boolean array over the profile points indicating selection.
        """
        if self._profile is not None and len(mask) == len(self._profile.distances):
            self._selected_indices = mask
        else:
            self._selected_indices = None

    @property
    def selected_mask(self) -> Optional[np.ndarray]:
        """Boolean mask of selected profile points, or ``None``."""
        return self._selected_indices

    @property
    def selected_count(self) -> int:
        if self._selected_indices is None:
            return 0
        return int(self._selected_indices.sum())

    def get_selected_original_indices(self) -> np.ndarray:
        """
        Return the indices of selected points in the ORIGINAL tile arrays.

        These indices can be passed to :meth:`TileManager.update_tile_classifications`.
        """
        if self._profile is None or self._selected_indices is None:
            return np.array([], dtype=np.int64)
        return self._profile.indices[self._selected_indices]

    # ── classification ─────────────────────────────────────────────

    def assign_class(self, new_class: int) -> bool:
        """
        Assign a new ASPRS class to the currently selected points.

        Returns ``True`` on success.
        """
        if self._tile_id is None:
            logger.warning("No tile open — cannot assign class")
            return False

        original_indices = self.get_selected_original_indices()
        if len(original_indices) == 0:
            logger.info("No points selected — nothing to classify")
            return False

        # Get old classes for undo
        if self._point_data is not None:
            old_classes = self._point_data["classification"][original_indices].copy()
        else:
            old_classes = np.zeros(len(original_indices), dtype=np.uint8)

        # Create and execute command
        cmd = ClassifyPointsCommand(
            tile_manager=self._tm,
            tile_id=self._tile_id,
            point_indices=original_indices,
            old_classes=old_classes,
            new_class=new_class,
        )
        self._edit_stack.push(cmd)

        # Refresh local point data
        self._point_data = self._tm.load_tile_points_full(self._tile_id)

        # Clear selection
        self._selected_indices = None

        logger.info(
            "Assigned class %d to %d point(s) in %s",
            new_class, len(original_indices), self._tile_id,
        )
        return True

    # ── undo / redo ────────────────────────────────────────────────

    def undo(self) -> Optional[str]:
        """
        Undo the last classification edit.

        Returns a description of the undone command, or ``None``.
        """
        cmd = self._edit_stack.undo()
        if cmd:
            # Refresh local data
            self._point_data = self._tm.load_tile_points_full(self._tile_id)
            return cmd.description
        return None

    def redo(self) -> Optional[str]:
        """
        Redo the last undone classification edit.
        """
        cmd = self._edit_stack.redo()
        if cmd:
            self._point_data = self._tm.load_tile_points_full(self._tile_id)
            return cmd.description
        return None

    @property
    def can_undo(self) -> bool:
        return self._edit_stack.can_undo

    @property
    def can_redo(self) -> bool:
        return self._edit_stack.can_redo

    @property
    def undo_stack_info(self) -> Tuple[int, int]:
        """Return ``(undo_count, redo_count)``."""
        return (self._edit_stack.undo_stack_size, self._edit_stack.redo_stack_size)
