"""Fetch host env: the PickCloth env (walker + crouch + hinged jaw) in procedural apartments.

Episodes, rewards and resets of the fetch task are owned by the high-level wrapper in the
parent repo (scripts/fetch/); this env only supplies the physics, the frozen-skill interface and
the cloth entities. The underlying episode length is effectively unbounded for that reason.
"""

from copy import deepcopy

from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_CLOTH_CFG

import os

from .home_layouts import ApartmentTerrainCfg, FixedPlanTerrainCfg
from .microduck_home_env_cfg import _soften_floor_contacts
from .microduck_pick_cloth_env_cfg import make_microduck_pick_cloth_env_cfg

NUM_CLOTHS = 3
APARTMENT_SIZE = (10.0, 10.0)

APARTMENT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=APARTMENT_SIZE,
    border_width=5.0,
    num_rows=6,
    num_cols=12,
    curriculum=False,
    sub_terrains={"apartment": ApartmentTerrainCfg(proportion=1.0)},
    add_lights=False,
)


def make_microduck_fetch_env_cfg(play: bool = False, num_cloths: int = NUM_CLOTHS):
    cfg = make_microduck_pick_cloth_env_cfg(play=play)
    entities = {k: v for k, v in cfg.scene.entities.items() if k != "cloth"}
    for i in range(num_cloths):
        entities[f"cloth_{i}"] = MICRODUCK_CLOTH_CFG
    cfg.scene.entities = entities
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = deepcopy(APARTMENT_TERRAINS_CFG)
    # MICRODUCK_FETCH_PLAN=<plan.json> puts every env in that one floor plan (held-out evaluation).
    plan = os.environ.get("MICRODUCK_FETCH_PLAN")
    if plan:
        gen = cfg.scene.terrain.terrain_generator
        gen.sub_terrains = {"plan": FixedPlanTerrainCfg(proportion=1.0, plan=plan)}
        gen.num_rows, gen.num_cols = 1, 1
    cfg.scene.spec_fn = _soften_floor_contacts
    # Walls and furniture add contacts; same headroom and solver settings as the rough walker.
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50
    cfg.episode_length_s = 1.0e5
    cfg.events.pop("push_robot", None)
    # Spawn on the station, facing anywhere.
    pose = cfg.events["reset_base"].params["pose_range"]
    pose["x"], pose["y"] = (-0.05, 0.05), (-0.05, 0.05)
    pose["yaw"] = (-3.14159, 3.14159)
    return cfg
