"""Velocity task on home-like floors: thresholds, thick rugs, cables and toys, slopes.

Terrain difficulty is sampled at random per patch instead of promoted by a curriculum: the
distance-based promotion needs the robot to walk more than half a patch (4 m) in an episode,
which a slow walker never does, so every robot used to stay on the easiest row.
"""

from copy import deepcopy

import mujoco as _mujoco
from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg, BoxRandomGridTerrainCfg
from mjlab.terrains.heightfield_terrains import HfPyramidSlopedTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from .home_terrains import HomeClutterTerrainCfg, HomeRugTerrainCfg, HomeThresholdTerrainCfg
from .microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

# Smoothness penalty held at this value from iteration 1000 on. At -1.0 it outweighed velocity
# tracking and the walker settled at 0.18 m/s against a 0.30 m/s command.
ACTION_RATE_FINAL = -0.4
FOOT_FRICTION_RANGE = (0.4, 1.5)  # tile and polished wood up to dry carpet

HOME_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum=False,
    sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=0.15),
        "thresholds": HomeThresholdTerrainCfg(proportion=0.20),
        "rugs": HomeRugTerrainCfg(proportion=0.25),
        "clutter": HomeClutterTerrainCfg(proportion=0.20),
        "uneven": BoxRandomGridTerrainCfg(
            proportion=0.10, grid_width=0.45, grid_height_range=(0.0, 0.010), platform_width=1.5
        ),
        "ramps": HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.03, 0.10), platform_width=2.0, vertical_scale=0.001
        ),
    },
    add_lights=False,
)

_DEFAULT_SOLREF = 0.02


def _soften_floor_contacts(spec: _mujoco.MjSpec) -> None:
    """Soften hard floor edges (they trigger NaN contact impulses) but keep the rugs' own softness."""
    for geom in spec.body("terrain").geoms:
        if abs(geom.solref[0] - _DEFAULT_SOLREF) < 1e-9:
            geom.solref = [0.04, 1.0]
            geom.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]


def make_microduck_home_env_cfg(play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=True)
    terrain_cfg = deepcopy(HOME_TERRAINS_CFG)
    if play:
        terrain_cfg.num_rows, terrain_cfg.num_cols = 5, 5
    cfg.scene.terrain.terrain_generator = terrain_cfg
    cfg.scene.spec_fn = _soften_floor_contacts
    cfg.curriculum.pop("terrain_levels", None)
    cfg.events["foot_friction"].params["ranges"] = FOOT_FRICTION_RANGE
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    for stage in stages:
        stage["weight"] = max(stage["weight"], ACTION_RATE_FINAL) if stage["weight"] < 0 else stage["weight"]
    return cfg
