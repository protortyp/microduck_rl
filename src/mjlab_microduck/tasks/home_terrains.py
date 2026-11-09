"""Sub-terrains that mimic what a home floor does to a 25 cm biped.

Every patch is 8x8 m with the spawn point at its centre, and keeps a flat platform around the
spawn so the robot never starts on top of an obstacle. `difficulty` in [0, 1] scales obstacle
height and count. Heights stay at or below about 2.5 cm because the robot lifts its feet 1-2 cm.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.primitive_terrains import make_plane
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

_FLOOR = (0.55, 0.45, 0.35, 1.0)
_RUG = (0.93, 0.92, 0.88, 1.0)
_THRESHOLD = (0.35, 0.3, 0.25, 1.0)
_CLUTTER = (0.8, 0.25, 0.2, 1.0)


def _quat_z(yaw: float):
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def _add_box(body, centre, half, yaw=0.0, rgba=_CLUTTER, solref=None, solimp=None, friction=None):
    geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(half),
        pos=list(centre),
        quat=_quat_z(yaw),
    )
    if solref is not None:
        geom.solref = list(solref)
    if solimp is not None:
        geom.solimp = list(solimp)
    if friction is not None:
        geom.friction = list(friction)
    return TerrainGeometry(geom=geom, color=rgba)


def _free_xy(rng, size, platform, margin):
    """Random (x, y) in the patch that stays outside the central flat platform."""
    while True:
        x, y = rng.uniform(margin, size[0] - margin), rng.uniform(margin, size[1] - margin)
        if max(abs(x - size[0] / 2), abs(y - size[1] / 2)) > platform / 2 + margin:
            return x, y


@dataclass(kw_only=True)
class HomeThresholdTerrainCfg(SubTerrainCfg):
    """Door thresholds: low bars across the floor, 4 mm up to 15 mm."""

    platform_width: float = 1.5
    num_bars: int = 6

    def function(self, difficulty, spec, rng):
        body = spec.body("terrain")
        geoms = [TerrainGeometry(geom=make_plane(body, self.size, 0.0, center_zero=False)[0], color=_FLOOR)]
        height = 0.004 + difficulty * 0.011
        for _ in range(self.num_bars):
            x, y = _free_xy(rng, self.size, self.platform_width, 0.3)
            length = rng.uniform(0.9, 3.0)
            width = rng.uniform(0.04, 0.08)
            yaw = rng.choice([0.0, np.pi / 2]) + rng.normal(0.0, 0.05)
            geoms.append(_add_box(body, (x, y, height / 2), (length / 2, width / 2, height / 2), yaw, _THRESHOLD))
        origin = (self.size[0] / 2, self.size[1] / 2, 0.0)
        return TerrainOutput(origin=np.array(origin), geometries=geoms)


@dataclass(kw_only=True)
class HomeEdgeFrameTerrainCfg(SubTerrainCfg):
    """A square frame of rug (soft) or floorboard (hard) around the spawn point.

    The robot starts in the frame's hole, so whatever it is commanded to do it meets a straight
    edge within a few seconds: up onto the frame, along it, and down off its outer edge. On the
    random rugs of HomeRugTerrainCfg an edge is met only now and then, and fetch walkers fell
    mostly at rug edges.

    With `heights` set, difficulty picks one of them (row r of a curriculum grid gets
    heights[r * len(heights) // num_rows]), which gives evaluations fixed heights per row.
    """

    height_range: tuple[float, float] = (0.005, 0.03)
    heights: tuple[float, ...] | None = None
    soft: bool = True
    hole_range: tuple[float, float] = (0.35, 0.6)  # half-width of the hole the robot starts in
    width_range: tuple[float, float] = (0.6, 1.0)  # width of the frame's sides

    def function(self, difficulty, spec, rng):
        body = spec.body("terrain")
        geoms = [TerrainGeometry(geom=make_plane(body, self.size, 0.0, center_zero=False)[0], color=_FLOOR)]
        if self.heights:
            h = self.heights[min(int(difficulty * len(self.heights)), len(self.heights) - 1)]
        else:
            h = self.height_range[0] + difficulty * (self.height_range[1] - self.height_range[0])
        hole, width = rng.uniform(*self.hole_range), rng.uniform(*self.width_range)
        yaw = rng.uniform(0, np.pi / 2)
        c, s = np.cos(yaw), np.sin(yaw)
        cx, cy = self.size[0] / 2, self.size[1] / 2
        mid, along = hole + width / 2, hole + width
        contact = {}
        if self.soft:
            contact = dict(solref=(0.06 + 0.04 * difficulty, 0.8), solimp=(0.7, 0.9, 0.003, 0.5, 2.0),
                           friction=(rng.uniform(0.9, 1.5), 0.005, 0.0001))
        # Two sides run the full outer length and two fit between them, so the corners do not overlap.
        for dx, dy, hx, hy in ((mid, 0, width / 2, along), (-mid, 0, width / 2, along),
                               (0, mid, hole, width / 2), (0, -mid, hole, width / 2)):
            centre = (cx + c * dx - s * dy, cy + s * dx + c * dy, h / 2)
            geoms.append(_add_box(body, centre, (hx, hy, h / 2), yaw, _RUG if self.soft else _THRESHOLD, **contact))
        origin = (cx, cy, 0.0)
        return TerrainOutput(origin=np.array(origin), geometries=geoms)


@dataclass(kw_only=True)
class HomeRugTerrainCfg(SubTerrainCfg):
    """Thick fluffy rugs: raised 5-25 mm with grippy, soft contact so a foot sinks in.

    The rug geoms get a softer solref than the floor, which is how the softening pass in the
    task config recognises them and leaves them alone.
    """

    platform_width: float = 1.5
    num_rugs: int = 3

    def function(self, difficulty, spec, rng):
        body = spec.body("terrain")
        geoms = [TerrainGeometry(geom=make_plane(body, self.size, 0.0, center_zero=False)[0], color=_FLOOR)]
        pile = 0.005 + difficulty * 0.02
        for _ in range(self.num_rugs):
            x, y = _free_xy(rng, self.size, self.platform_width, 0.5)
            half = (rng.uniform(0.6, 1.3), rng.uniform(0.6, 1.3), pile / 2)
            geoms.append(_add_box(
                body, (x, y, pile / 2), half, rng.uniform(0, np.pi), _RUG,
                solref=(0.06 + 0.04 * difficulty, 0.8), solimp=(0.7, 0.9, 0.003, 0.5, 2.0),
                friction=(rng.uniform(0.9, 1.5), 0.005, 0.0001),
            ))
        origin = (self.size[0] / 2, self.size[1] / 2, 0.0)
        return TerrainOutput(origin=np.array(origin), geometries=geoms)


@dataclass(kw_only=True)
class HomeClutterTerrainCfg(SubTerrainCfg):
    """Cables (thin, long) and toys (small blocks) scattered on the floor."""

    platform_width: float = 1.2
    max_items: int = 40

    def function(self, difficulty, spec, rng):
        body = spec.body("terrain")
        geoms = [TerrainGeometry(geom=make_plane(body, self.size, 0.0, center_zero=False)[0], color=_FLOOR)]
        count = int(8 + difficulty * (self.max_items - 8))
        for _ in range(count):
            x, y = _free_xy(rng, self.size, self.platform_width, 0.2)
            if rng.random() < 0.5:  # cable
                h = rng.uniform(0.004, 0.008) * (0.5 + difficulty)
                half = (rng.uniform(0.3, 1.0), h / 2 + 0.001, h / 2)
            else:  # toy
                h = rng.uniform(0.01, 0.025) * (0.4 + 0.6 * difficulty)
                half = (rng.uniform(0.01, 0.04), rng.uniform(0.01, 0.04), h / 2)
            geoms.append(_add_box(body, (x, y, half[2]), half, rng.uniform(0, np.pi)))
        origin = (self.size[0] / 2, self.size[1] / 2, 0.0)
        return TerrainOutput(origin=np.array(origin), geometries=geoms)
