"""Procedural apartments as terrain patches: rooms, doorways, furniture, rugs, thresholds, cables.

Each 10x10 m patch is one apartment. Rooms come from recursive splits of the floor; every split
wall gets one doorway, so all rooms stay connected. The robot spawns at the station, a free spot
in one room. Alongside the MuJoCo geometry, every patch records a 2D occupancy grid (walls and
furniture) in LAYOUTS, which the high-level fetch env uses for line of sight, camera coverage and
path distances. Patches are generated row-major, so the patch at (row, col) is entry
``row * num_cols + col`` of the most recent ``len == num_rows * num_cols`` block.
"""

from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np
from mjlab.terrains.primitive_terrains import make_plane
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

CELL = 0.1  # m, occupancy grid resolution
WALL_T = 0.1
WALL_H = 0.5  # m, tall enough to block the head camera (about 0.22 m)
DOOR_W = (0.8, 1.0)
MIN_ROOM = 2.0
STATION_CLEARANCE = 0.8

_FLOOR = (0.62, 0.52, 0.42, 1.0)
_WALL = (0.92, 0.9, 0.86, 1.0)
_FURNITURE = (0.45, 0.5, 0.58, 1.0)
_RUG = (0.95, 0.94, 0.9, 1.0)
_THRESHOLD = (0.4, 0.33, 0.27, 1.0)
_CABLE = (0.15, 0.15, 0.15, 1.0)


@dataclass
class Layout:
    occupancy: np.ndarray  # (H, W) uint8, 1 = wall or furniture, row index = y / CELL
    station: tuple[float, float]  # patch-local (x, y) of the spawn / station
    rooms: list = field(default_factory=list)  # (x0, y0, x1, y1) per room, patch-local
    rugs: list = field(default_factory=list)  # (x, y, half_x, half_y, yaw)


LAYOUTS: list[Layout] = []


def _box(body, centre, half, rgba, yaw=0.0, **contact):
    geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(centre),
                         quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    for k, v in contact.items():
        setattr(geom, k, list(v))
    return TerrainGeometry(geom=geom, color=rgba)


def _mark(occ, x0, y0, x1, y1):
    i0, i1 = int(np.floor(y0 / CELL)), int(np.ceil(y1 / CELL))
    j0, j1 = int(np.floor(x0 / CELL)), int(np.ceil(x1 / CELL))
    occ[max(i0, 0):max(i1, 0), max(j0, 0):max(j1, 0)] = 1


def _reachable(occ, start):
    h, w = occ.shape
    seen = np.zeros_like(occ, dtype=bool)
    q = deque([start])
    seen[start] = True
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w and not seen[a, b] and not occ[a, b]:
                seen[a, b] = True
                q.append((a, b))
    return seen


def _inflate(occ, r=1):
    """Binary dilation by r cells; the patch edge counts as wall (no wrap-around)."""
    h, w = occ.shape
    pad = np.pad(occ, r, constant_values=1)
    out = np.zeros_like(occ)
    for di in range(2 * r + 1):
        for dj in range(2 * r + 1):
            out |= pad[di:di + h, dj:dj + w]
    return out


@dataclass(kw_only=True)
class ApartmentTerrainCfg(SubTerrainCfg):
    """One apartment per patch. `difficulty` scales room count, clutter and obstacle heights."""

    rooms_range: tuple[int, int] = (3, 6)
    margin: float = 0.2  # m between the patch edge and the outer wall

    def function(self, difficulty, spec, rng):
        body = spec.body("terrain")
        sx, sy = self.size
        geoms = [TerrainGeometry(geom=make_plane(body, self.size, 0.0, center_zero=False)[0], color=_FLOOR)]
        occ = np.zeros((int(round(sy / CELL)), int(round(sx / CELL))), dtype=np.uint8)
        m = self.margin
        inner = (m + WALL_T, m + WALL_T, sx - m - WALL_T, sy - m - WALL_T)

        def wall(x0, y0, x1, y1):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            geoms.append(_box(body, (cx, cy, WALL_H / 2), ((x1 - x0) / 2, (y1 - y0) / 2, WALL_H / 2), _WALL))
            _mark(occ, x0, y0, x1, y1)

        # Outer walls.
        wall(m, m, sx - m, m + WALL_T)
        wall(m, sy - m - WALL_T, sx - m, sy - m)
        wall(m, m, m + WALL_T, sy - m)
        wall(sx - m - WALL_T, m, sx - m, sy - m)

        # Rooms by recursive splits; each split wall keeps one doorway.
        target = int(round(self.rooms_range[0] + difficulty * (self.rooms_range[1] - self.rooms_range[0])))
        rooms = [inner]
        doors = []
        for _ in range(40):
            if len(rooms) >= target:
                break
            splittable = [i for i, r in enumerate(rooms)
                          if max(r[2] - r[0], r[3] - r[1]) >= 2 * MIN_ROOM + WALL_T]
            if not splittable:
                break
            k = max(splittable, key=lambda i: (rooms[i][2] - rooms[i][0]) * (rooms[i][3] - rooms[i][1]))
            x0, y0, x1, y1 = rooms[k]
            vertical = (x1 - x0) >= (y1 - y0)
            span = (x1 - x0) if vertical else (y1 - y0)
            cut = rng.uniform(MIN_ROOM, span - MIN_ROOM - WALL_T)
            door = rng.uniform(*DOOR_W)
            if vertical:
                wx = x0 + cut
                d0 = rng.uniform(y0 + 0.2, y1 - 0.2 - door)
                wall(wx, y0, wx + WALL_T, d0)
                wall(wx, d0 + door, wx + WALL_T, y1)
                doors.append((wx + WALL_T / 2, d0 + door / 2, door, True))
                rooms[k:k + 1] = [(x0, y0, wx, y1), (wx + WALL_T, y0, x1, y1)]
            else:
                wy = y0 + cut
                d0 = rng.uniform(x0 + 0.2, x1 - 0.2 - door)
                wall(x0, wy, d0, wy + WALL_T)
                wall(d0 + door, wy, x1, wy + WALL_T)
                doors.append((d0 + door / 2, wy + WALL_T / 2, door, False))
                rooms[k:k + 1] = [(x0, y0, x1, wy), (x0, wy + WALL_T, x1, y1)]

        # Station: a free spot near the centre of a random room.
        r = rooms[int(rng.integers(len(rooms)))]
        station = ((r[0] + r[2]) / 2 + rng.uniform(-0.3, 0.3), (r[1] + r[3]) / 2 + rng.uniform(-0.3, 0.3))

        def near_door_or_station(x0, y0, x1, y1, pad):
            for dx, dy, dw, _ in doors:
                if x0 - pad < dx < x1 + pad and y0 - pad < dy < y1 + pad:
                    return True
            return (x0 - STATION_CLEARANCE < station[0] < x1 + STATION_CLEARANCE
                    and y0 - STATION_CLEARANCE < station[1] < y1 + STATION_CLEARANCE)

        # Furniture against room walls (sofas, cabinets, beds), kept clear of doorways.
        furniture = []
        for r in rooms:
            for _ in range(int(rng.integers(0, 3))):
                w, d = rng.uniform(0.5, 1.6), rng.uniform(0.4, 0.8)
                side = int(rng.integers(4))
                if side in (0, 1):  # against bottom / top wall
                    if r[2] - r[0] < w + 0.4:
                        continue
                    x = rng.uniform(r[0] + 0.1, r[2] - 0.1 - w)
                    box = (x, r[1], x + w, r[1] + d) if side == 0 else (x, r[3] - d, x + w, r[3])
                else:
                    if r[3] - r[1] < w + 0.4:
                        continue
                    y = rng.uniform(r[1] + 0.1, r[3] - 0.1 - w)
                    box = (r[0], y, r[0] + d, y + w) if side == 2 else (r[2] - d, y, r[2], y + w)
                if near_door_or_station(*box, pad=0.7):
                    continue
                if any(not (box[2] + 0.1 < f[0] or f[2] + 0.1 < box[0] or box[3] + 0.1 < f[1] or f[3] + 0.1 < box[1])
                       for f in furniture):
                    continue
                furniture.append(box)
        for box in furniture:
            trial = occ.copy()
            _mark(trial, *box)
            si, sj = int(station[1] / CELL), int(station[0] / CELL)
            if _inflate(trial)[si, sj]:
                continue
            before = _reachable(_inflate(occ), (si, sj)).sum()
            after = _reachable(_inflate(trial), (si, sj)).sum()
            # Drop furniture that walls off part of the floor rather than just occupying it.
            footprint = (trial - occ).sum()
            if before - after > footprint * 2.5 + 40:
                continue
            occ[:] = trial
            x0, y0, x1, y1 = box
            h = rng.uniform(0.3, 0.5)
            geoms.append(_box(body, ((x0 + x1) / 2, (y0 + y1) / 2, h / 2), ((x1 - x0) / 2, (y1 - y0) / 2, h / 2), _FURNITURE))

        # Thick rugs (soft, grippy, raised 4-15 mm), not over the station.
        rugs = []
        for r in rooms:
            if rng.random() < 0.5:
                continue
            hx, hy = rng.uniform(0.5, 1.1), rng.uniform(0.4, 0.8)
            if r[2] - r[0] < 2 * hx + 0.2 or r[3] - r[1] < 2 * hy + 0.2:
                continue
            cx, cy = rng.uniform(r[0] + hx + 0.05, r[2] - hx - 0.05), rng.uniform(r[1] + hy + 0.05, r[3] - hy - 0.05)
            if abs(cx - station[0]) < hx + 0.3 and abs(cy - station[1]) < hy + 0.3:
                continue
            pile = 0.004 + difficulty * 0.011  # compressed pile height; the walker lifts its feet 1-2 cm
            geoms.append(_box(body, (cx, cy, pile / 2), (hx, hy, pile / 2), _RUG,
                              solref=(0.06 + 0.04 * difficulty, 0.8), solimp=(0.7, 0.9, 0.003, 0.5, 2.0),
                              friction=(rng.uniform(0.9, 1.5), 0.005, 0.0001)))
            rugs.append((cx, cy, hx, hy, 0.0))

        # Door thresholds (4-12 mm) in some doorways.
        for dx, dy, dw, vertical in doors:
            if rng.random() < 0.5:
                h = 0.004 + difficulty * 0.008
                half = (0.03, dw / 2, h / 2) if vertical else (dw / 2, 0.03, h / 2)
                geoms.append(_box(body, (dx, dy, h / 2), half, _THRESHOLD))

        # A few cables on the floor.
        for _ in range(int(rng.integers(0, 3 + int(4 * difficulty)))):
            r = rooms[int(rng.integers(len(rooms)))]
            cx, cy = rng.uniform(r[0] + 0.3, r[2] - 0.3), rng.uniform(r[1] + 0.3, r[3] - 0.3)
            if np.hypot(cx - station[0], cy - station[1]) < STATION_CLEARANCE:
                continue
            h = rng.uniform(0.004, 0.008)
            half_len, yaw = rng.uniform(0.3, 0.8), rng.uniform(0, np.pi)
            ends = [(cx + sgn * half_len * np.cos(yaw), cy + sgn * half_len * np.sin(yaw)) for sgn in (-1, 1)]
            # Both ends inside the same room, so a cable never runs through a wall or doorway.
            if not all(r[0] + 0.1 < x < r[2] - 0.1 and r[1] + 0.1 < y < r[3] - 0.1 for x, y in ends):
                continue
            geoms.append(_box(body, (cx, cy, h / 2), (half_len, 0.004, h / 2), _CABLE, yaw=yaw))

        LAYOUTS.append(Layout(occupancy=occ, station=station, rooms=rooms, rugs=rugs))
        return TerrainOutput(origin=np.array([station[0], station[1], 0.0]), geometries=geoms)
