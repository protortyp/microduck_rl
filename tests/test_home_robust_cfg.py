import mujoco
import numpy as np
import pytest

from mjlab_microduck.tasks.home_terrains import HomeEdgeFrameTerrainCfg
from mjlab_microduck.tasks.microduck_home_env_cfg import (
    ROBUST_ANG_VEL_Z,
    ROBUST_EDGE_PROPORTION,
    make_microduck_home_robust_env_cfg,
)


def test_robust_cfg_terrain_mix_and_commands():
    cfg = make_microduck_home_robust_env_cfg()
    subs = cfg.scene.terrain.terrain_generator.sub_terrains
    assert sum(t.proportion for t in subs.values()) == pytest.approx(1.0)
    assert subs["edge_soft"].proportion + subs["edge_hard"].proportion == pytest.approx(ROBUST_EDGE_PROPORTION)
    assert subs["edge_soft"].soft and not subs["edge_hard"].soft
    twist = cfg.commands["twist"]
    assert twist.ranges.ang_vel_z == ROBUST_ANG_VEL_Z
    assert twist.resampling_time_range[0] < 1.0  # commands switch mid-stride
    assert cfg.rewards["track_linear_velocity_wide"].weight > 0
    assert cfg.rewards["track_angular_velocity_tight"].weight > 0


def _frame(difficulty, **kw):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    t = HomeEdgeFrameTerrainCfg(proportion=1.0, size=(8.0, 8.0), **kw)
    out = t.function(difficulty, spec, np.random.default_rng(0))
    return out, [g.geom for g in out.geometries[1:]]


def test_edge_frame_fixed_heights_by_row():
    heights = (0.005, 0.015, 0.03)
    for row, h in enumerate(heights):
        _, boxes = _frame((row + 0.5) / len(heights), heights=heights)
        assert len(boxes) == 4
        for b in boxes:
            assert 2 * b.size[2] == pytest.approx(h)


def test_edge_frame_leaves_spawn_clear():
    out, boxes = _frame(0.5, height_range=(0.01, 0.01), hole_range=(0.4, 0.4))
    spawn = np.asarray(out.origin[:2])
    for b in boxes:
        # Distance from the spawn to the nearest point of the rotated box.
        q = b.quat
        yaw = 2 * np.arctan2(q[3], q[0])
        d = spawn - np.asarray(b.pos[:2])
        local = np.array([np.cos(yaw) * d[0] + np.sin(yaw) * d[1], -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]])
        gap = np.maximum(np.abs(local) - np.asarray(b.size[:2]), 0)
        assert np.linalg.norm(gap) >= 0.4 - 1e-6



def test_robust_cfg_raises_swing_height():
    from mjlab_microduck.tasks.microduck_home_env_cfg import ROBUST_SWING_HEIGHT

    cfg = make_microduck_home_robust_env_cfg()
    for name in ("foot_clearance", "foot_swing_height"):
        assert cfg.rewards[name].params["target_height"] == ROBUST_SWING_HEIGHT
