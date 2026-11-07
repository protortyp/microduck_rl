"""Microduck PickCloth host env: the GroundPick env plus a cloth patch on the floor.

The pick-up itself is not learned end to end here. The walking and crouch policies
share the 61-D observation / 14-D action contract, so scripts/pick_cloth.py swaps
between them inside this one env and a scripted state machine decides when. The
cloth is placed and moved by that script, not by an env event.
"""

from mjlab_microduck.robot.microduck_constants import MICRODUCK_CLOTH_CFG

import torch
from mjlab.managers.observation_manager import ObservationTermCfg

from .microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg


def scripted_head_command(env) -> torch.Tensor:
    """Head-pose command slot [neck_pitch, head_pitch, head_yaw, head_roll] set by the caller.

    The caller stores a (num_envs, 4) tensor in `env.head_command_override`; until then the head
    holds its home pose, as in the crouch task.
    """
    cmd = getattr(env, "head_command_override", None)
    return cmd if cmd is not None else torch.zeros(env.num_envs, 4, device=env.device)


def make_microduck_pick_cloth_env_cfg(**kwargs):
    cfg = make_microduck_ground_pick_env_cfg(**kwargs)
    # The robot must stay the first entity: reset events write its root state at qpos[:, 0:7].
    cfg.scene.entities = {**cfg.scene.entities, "cloth": MICRODUCK_CLOTH_CFG}
    # The crouch task randomises a payload force on the mouth; the walker never saw it
    # and the cloth weighs almost nothing, so the host env leaves it out.
    cfg.events.pop("sample_mouth_payload", None)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(func=scripted_head_command)
    # The jaw actuator would otherwise join the policy's action vector.
    cfg.actions["joint_pos"].actuator_names = (r"^(?!passive_).*",)
    # The walker was trained with at most 20 ms of IMU lag; the crouch task's 60 ms
    # worst case is enough to topple it, and the crouch policy tolerates the shorter lag.
    for term in ("base_ang_vel", "projected_gravity"):
        cfg.observations["actor"].terms[term].delay_max_lag = 1
    return cfg
