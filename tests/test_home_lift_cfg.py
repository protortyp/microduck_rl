import pytest
import torch
from types import SimpleNamespace

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_home_env_cfg import (
    LIFT_TARGET,
    make_microduck_home_lift_env_cfg,
)


def test_lift_cfg_rewards_and_critic_scan():
    cfg = make_microduck_home_lift_env_cfg()
    r = cfg.rewards
    assert "foot_clearance" not in r and "foot_swing_height" not in r
    assert r["swing_foot_height"].weight > 0 and r["air_time"].weight > 0
    assert r["foot_stumble"].weight < 0  # cost >= 0 needs a negative weight
    assert r["air_time"].func is mdp.feet_air_time_lifted
    assert r["swing_foot_height"].params["target_height"] == LIFT_TARGET
    assert "height_scan" in cfg.observations["critic"].terms
    assert "height_scan" not in cfg.observations["actor"].terms  # actor stays blind and 61D
    assert any(s.name == "terrain_scan" for s in cfg.scene.sensors)


def _env(found, air, heights, cmd=(0.2, 0.0, 0.0), force=None):
    contact = SimpleNamespace(data=SimpleNamespace(found=torch.tensor(found), current_air_time=torch.tensor(air),
                                                   force=force))
    scan = SimpleNamespace(data=SimpleNamespace(heights=torch.tensor(heights)))
    cm = SimpleNamespace(get_command=lambda _: torch.tensor([cmd], dtype=torch.float32))
    return SimpleNamespace(scene={"c": contact, "h": scan}, command_manager=cm, extras={"log": {}})


def test_swing_foot_height_pays_lift_up_to_target_and_only_early_in_swing():
    kw = dict(sensor_name="c", height_sensor_name="h", target_height=0.03, command_name="twist")
    # left foot swinging at 1.5 cm, right foot on the ground
    assert mdp.swing_foot_height(_env([[0, 1]], [[0.1, 0.0]], [[0.015, 0.0]]), **kw).item() == pytest.approx(0.5)
    # above the target pays no more
    assert mdp.swing_foot_height(_env([[0, 1]], [[0.1, 0.0]], [[0.06, 0.0]]), **kw).item() == pytest.approx(1.0)
    # a foot held up longer than max_air_time pays nothing
    assert mdp.swing_foot_height(_env([[0, 1]], [[0.5, 0.0]], [[0.03, 0.0]]), **kw).item() == 0.0
    # no command, no pay
    assert mdp.swing_foot_height(_env([[0, 1]], [[0.1, 0.0]], [[0.03, 0.0]], cmd=(0, 0, 0)), **kw).item() == 0.0


def test_foot_stumble_flags_horizontal_pushes_only():
    force = torch.tensor([[[3.0, 0.0, 0.2], [0.5, 0.0, 2.0]]])  # toe against a face; normal stance
    env = _env([[1, 1]], [[0.0, 0.0]], [[0.0, 0.0]], force=force)
    assert mdp.foot_stumble(env, sensor_name="c").item() == 1.0
