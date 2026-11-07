"""Microduck PickCloth host env: the GroundPick env plus a cloth patch on the floor.

The pick-up itself is not learned end to end here. The walking and crouch policies
share the 61-D observation / 14-D action contract, so scripts/pick_cloth.py swaps
between them inside this one env and a scripted state machine decides when. The
cloth is placed and moved by that script, not by an env event.
"""

from mjlab_microduck.robot.microduck_constants import MICRODUCK_CLOTH_CFG

from .microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg


def make_microduck_pick_cloth_env_cfg(**kwargs):
    cfg = make_microduck_ground_pick_env_cfg(**kwargs)
    # The robot must stay the first entity: reset events write its root state at qpos[:, 0:7].
    cfg.scene.entities = {**cfg.scene.entities, "cloth": MICRODUCK_CLOTH_CFG}
    # The crouch task randomises a payload force on the mouth; the walker never saw it
    # and the cloth weighs almost nothing, so the host env leaves it out.
    cfg.events.pop("sample_mouth_payload", None)
    # The walker was trained with at most 20 ms of IMU lag; the crouch task's 60 ms
    # worst case is enough to topple it, and the crouch policy tolerates the shorter lag.
    for term in ("base_ang_vel", "projected_gravity"):
        cfg.observations["actor"].terms[term].delay_max_lag = 1
    return cfg
