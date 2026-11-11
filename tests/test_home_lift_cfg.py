import math
import pytest
import torch
from types import SimpleNamespace

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_home_env_cfg import (
    LIFT_FAST_LIN_VEL_X,
    LIFT_TARGET,
    make_microduck_home_lift_agile_env_cfg,
    make_microduck_home_lift_env_cfg,
    make_microduck_home_lift_pace_env_cfg,
    make_microduck_home_lift_stance_env_cfg,
    make_microduck_home_lift_turn_env_cfg,
    make_microduck_home_lift_fast_env_cfg,
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


def test_lift_fast_keeps_lift_terms_and_widens_forward_commands():
    cfg = make_microduck_home_lift_fast_env_cfg()
    assert cfg.commands["twist"].ranges.lin_vel_x == LIFT_FAST_LIN_VEL_X
    assert cfg.commands["twist"].rel_forward_envs > 0.3
    assert cfg.rewards["air_time"].func is mdp.feet_air_time_lifted
    assert "height_scan" in cfg.observations["critic"].terms


def test_lift_agile_adds_wide_yaw_term_and_turn_in_place_share():
    cfg, fast = make_microduck_home_lift_agile_env_cfg(), make_microduck_home_lift_fast_env_cfg()
    r = cfg.rewards
    assert r["track_angular_velocity_wide"].params["std"] > r["track_angular_velocity"].params["std"]
    assert r["track_angular_velocity_tight"].params["std"] < r["track_angular_velocity"].params["std"]
    assert cfg.commands["twist"].rel_turn_in_place_envs > fast.commands["twist"].rel_turn_in_place_envs
    assert cfg.commands["twist"].ranges.lin_vel_x == LIFT_FAST_LIN_VEL_X


def test_averaged_yaw_tracking_ignores_the_waddle_and_prices_the_mean():
    cfg = make_microduck_home_lift_turn_env_cfg()
    term_cfg = cfg.rewards["track_yaw_rate_averaged"]
    assert term_cfg.func is mdp.track_yaw_rate_averaged

    def run(mean_wz):
        robot = SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=torch.zeros(1, 3)))
        cm = SimpleNamespace(get_command=lambda _: torch.tensor([[0.0, 0.0, 1.5]]))
        env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.02, scene={"robot": robot}, command_manager=cm,
                              extras={"log": {}})
        term = mdp.track_yaw_rate_averaged(SimpleNamespace(params={"tau": 0.4}), env)
        for i in range(400):
            # +-1.1 rad/s at a 0.4 s stride on top of the mean turn
            robot.data.root_link_ang_vel_b[0, 2] = mean_wz + 1.1 * torch.sin(torch.tensor(i * 0.02 * 6.283 / 0.4))
            r = term(env, std=1.0, command_name="twist")
        return r.item(), env.extras["log"]["Metrics/yaw_rate_shortfall"].item()

    on_target, short_on = run(1.5)
    third, short_third = run(0.5)
    assert on_target > 0.8 and third < 0.5
    assert abs(short_on) < 0.4 and short_third == pytest.approx(1.0, abs=0.4)


def test_stance_adds_linear_posture_costs():
    cfg = make_microduck_home_lift_stance_env_cfg()
    r = cfg.rewards
    for name in ("stand_posture", "hip_splay", "trunk_low"):
        assert r[name].weight < 0
    assert r["stand_posture"].params["standing_only"] and not r["hip_splay"].params["standing_only"]
    assert "track_yaw_rate_averaged" in r and r["air_time"].func is mdp.feet_air_time_lifted

    data = SimpleNamespace(joint_pos=torch.tensor([[0.5, -0.5, 0.0]]), default_joint_pos=torch.zeros(1, 3))
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1])

    def env(cmd):
        cm = SimpleNamespace(get_command=lambda _: torch.tensor([cmd], dtype=torch.float32))
        return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, command_manager=cm)

    kw = dict(command_name="twist", asset_cfg=asset_cfg)
    assert mdp.leg_posture_l1(env((0, 0, 0)), standing_only=True, **kw).item() == pytest.approx(0.5)
    assert mdp.leg_posture_l1(env((0.2, 0, 0)), standing_only=True, **kw).item() == 0.0
    assert mdp.leg_posture_l1(env((0.2, 0, 0)), standing_only=False, **kw).item() == pytest.approx(0.5)


def test_pace_tracks_stride_averaged_forward_speed():
    cfg = make_microduck_home_lift_pace_env_cfg()
    term = cfg.rewards["track_forward_speed_averaged"]
    assert term.weight > 0 and "stand_posture" in cfg.rewards

    data = SimpleNamespace(root_link_lin_vel_b=torch.tensor([[0.2, 0.0, 0.0]]))
    cm = SimpleNamespace(get_command=lambda _: torch.tensor([[0.4, 0.0, 0.0]]))
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=0.02, extras={"log": {}},
                          scene={"robot": SimpleNamespace(data=data)}, command_manager=cm)
    f = mdp.track_forward_speed_averaged(term, env)
    kw = dict(std=0.15, command_name="twist", asset_cfg=SimpleNamespace(name="robot"))
    first = f(env, **kw).item()
    for _ in range(200):
        last = f(env, **kw).item()
    # The average converges to the realised 0.2 m/s: the 0.2 m/s shortfall is logged and paid for.
    assert last > first
    assert env.extras["log"]["Metrics/forward_speed_shortfall"].item() == pytest.approx(0.2, abs=1e-3)
    assert last == pytest.approx(math.exp(-(0.2 / 0.15) ** 2), abs=1e-3)


def test_servo_effort_excess_prices_only_what_exceeds_the_limits():
    data = SimpleNamespace(joint_vel=torch.tensor([[2.0, -8.0], [1.0, 1.0]]),
                           actuator_force=torch.tensor([[0.2, 0.6], [0.1, 0.1]]))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, extras={"log": {}})
    cost = mdp.servo_effort_excess(env, speed_limit=5.0, torque_limit=0.5, asset_cfg=SimpleNamespace(name="robot"))
    assert cost[0].item() == pytest.approx(3.0**2 + 100.0 * 0.1**2)
    assert cost[1].item() == 0.0
    assert env.extras["log"]["Metrics/peak_joint_speed"].item() == pytest.approx(4.5)
