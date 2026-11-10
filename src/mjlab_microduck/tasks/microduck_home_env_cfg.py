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
from mjlab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

from . import mdp as microduck_mdp

from .home_terrains import (
    HomeClutterTerrainCfg,
    HomeEdgeFrameTerrainCfg,
    HomeRugTerrainCfg,
    HomeThresholdTerrainCfg,
)
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


def _home_floors(cfg, play):
    terrain_cfg = deepcopy(HOME_TERRAINS_CFG)
    if play:
        terrain_cfg.num_rows, terrain_cfg.num_cols = 5, 5
    cfg.scene.terrain.terrain_generator = terrain_cfg
    cfg.scene.spec_fn = _soften_floor_contacts
    cfg.curriculum.pop("terrain_levels", None)
    cfg.events["foot_friction"].params["ranges"] = FOOT_FRICTION_RANGE


def make_microduck_home_env_cfg(play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=True)
    _home_floors(cfg, play)
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    for stage in stages:
        stage["weight"] = max(stage["weight"], ACTION_RATE_FINAL) if stage["weight"] < 0 else stage["weight"]
    return cfg


# The fetch executive walks forward almost all the time; the base walker, trained on a symmetric
# +-0.4 m/s range, realises at most 0.25 m/s and falls more the harder it is pushed. These settings
# spend more samples on fast forward walking and add a wide speed-tracking term: the base term's
# Gaussian (std 0.15) is flat at the 0.15-0.3 m/s shortfall a fast command starts from, so on its
# own it pays nothing for walking faster and a walker trained with a tighter std slowed to a shuffle.
FAST_LIN_VEL_X = (-0.2, 0.45)
FAST_FORWARD_FRACTION = 0.3
FAST_WIDE_TRACK_STD = 0.35
FAST_WIDE_TRACK_WEIGHT = 2.0


def make_microduck_home_fast_env_cfg(play: bool = False):
    cfg = make_microduck_home_env_cfg(play=play)
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = FAST_LIN_VEL_X
    twist.rel_forward_envs = FAST_FORWARD_FRACTION
    wide = deepcopy(cfg.rewards["track_linear_velocity"])
    wide.weight = FAST_WIDE_TRACK_WEIGHT
    wide.params["std"] = FAST_WIDE_TRACK_STD
    cfg.rewards["track_linear_velocity_wide"] = wide
    return cfg


# Turning is what limits the fetch task: the base walker realises about half of a yaw-rate command
# (0.49 rad/s at 1.0, 1.47 at 3.0) and capping the executive's turn command at 1.0 rad/s halves the
# deliveries. The base yaw term (std 0.71) pays 60% for a 0.5 rad/s shortfall, so undershooting
# costs little; an extra term with std 0.4 prices it, while the base term keeps the gradient for
# large errors.
TURN_ANG_VEL_Z = (-2.0, 2.0)
TURN_TRACK_STD = 0.4
TURN_TRACK_WEIGHT = 2.0


def make_microduck_home_turn_env_cfg(play: bool = False):
    cfg = make_microduck_home_env_cfg(play=play)
    cfg.commands["twist"].ranges.ang_vel_z = TURN_ANG_VEL_Z
    tight = deepcopy(cfg.rewards["track_angular_velocity"])
    tight.weight = TURN_TRACK_WEIGHT
    tight.params["std"] = TURN_TRACK_STD
    cfg.rewards["track_angular_velocity_tight"] = tight
    return cfg


# Robust home walker, the base skill for the fetch task. Measured gaps of home-walk3 inside fetch:
# it tips at 1.2-1.4 cm rug edges (most held-out falls happen there), and the fetch executive
# commands turn rates up to 3 rad/s and changes the command every 0.2 s, while the walker was
# trained on +-1 rad/s held for several seconds. These settings put a straight edge next to every
# spawn in a third of the patches, cover the executive's command range, and resample commands
# often enough that switching mid-stride is ordinary experience.
ROBUST_EDGE_PROPORTION = 0.35  # split between soft (rug) and hard (floorboard) frames
ROBUST_SOFT_EDGE_HEIGHT = (0.005, 0.03)
ROBUST_HARD_EDGE_HEIGHT = (0.005, 0.025)
ROBUST_LIN_VEL_X = (-0.2, 0.4)
ROBUST_ANG_VEL_Z = (-2.0, 2.0)
ROBUST_FORWARD_FRACTION = 0.3
# Uniform per env, so about a fifth of the commands last under 1.5 s.
ROBUST_RESAMPLE_S = (0.3, 6.0)
# The policy is blind (no height scan, as on the robot), so it clears an edge only by lifting every
# swing foot above it. home-walk3, trained to a 2 cm swing height, fell at 42-59% of 1.5 cm edges
# and 81-96% of 2 cm edges (docs/skills/walker/home-walk3_18000.json).
ROBUST_SWING_HEIGHT = 0.03


def _robust(cfg):
    gen = cfg.scene.terrain.terrain_generator
    others = sum(t.proportion for t in gen.sub_terrains.values())
    for t in gen.sub_terrains.values():
        t.proportion *= (1.0 - ROBUST_EDGE_PROPORTION) / others
    half = ROBUST_EDGE_PROPORTION / 2
    gen.sub_terrains["edge_soft"] = HomeEdgeFrameTerrainCfg(
        proportion=half, soft=True, height_range=ROBUST_SOFT_EDGE_HEIGHT)
    gen.sub_terrains["edge_hard"] = HomeEdgeFrameTerrainCfg(
        proportion=half, soft=False, height_range=ROBUST_HARD_EDGE_HEIGHT)

    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = ROBUST_LIN_VEL_X
    twist.ranges.ang_vel_z = ROBUST_ANG_VEL_Z
    twist.rel_forward_envs = ROBUST_FORWARD_FRACTION
    twist.resampling_time_range = ROBUST_RESAMPLE_S
    for name in ("foot_clearance", "foot_swing_height"):
        cfg.rewards[name].params["target_height"] = ROBUST_SWING_HEIGHT
    # Same wide speed term and tight yaw term as the fast and turn variants: the base terms pay
    # most of their reward for undershooting fast or sharp commands.
    wide = deepcopy(cfg.rewards["track_linear_velocity"])
    wide.weight, wide.params["std"] = FAST_WIDE_TRACK_WEIGHT, FAST_WIDE_TRACK_STD
    cfg.rewards["track_linear_velocity_wide"] = wide
    tight = deepcopy(cfg.rewards["track_angular_velocity"])
    tight.weight, tight.params["std"] = TURN_TRACK_WEIGHT, TURN_TRACK_STD
    cfg.rewards["track_angular_velocity_tight"] = tight
    return cfg


def make_microduck_home_robust_env_cfg(play: bool = False):
    return _robust(make_microduck_home_env_cfg(play=play))


# Foot lift for the robust walker. Measured on home-robust1: walking straight on
# flat floor the swing apex is ~19 mm (p95 26 mm), yet a quarter to half of the
# robots fall at 1.5 cm edges, because the foot is low early and late in each
# swing, when the toe meets the edge. The base foot terms cannot change that
# (see mdp.swing_foot_height), so these replace them:
# - a per-step lift reward on every swing foot, worth up to ~0.8/s against the
#   ~1.5/s action-rate penalty, so a whole-swing lift is affordable;
# - air time paid in proportion to how high the swing got, and only while the
#   trunk moves;
# - a stumble cost for toes pushing against edge faces.
# The critic also gets a terrain height scan (privileged, training only): with
# it, value estimates see an edge coming and falls there stop being surprises.
# The actor stays blind and 61D.
LIFT_TARGET = 0.03
LIFT_WEIGHT = 1.0
LIFT_MAX_AIR_TIME = 0.3
AIR_TIME_WEIGHT = 3.0
STUMBLE_WEIGHT = -1.0
CRITIC_SCAN_SIZE = (0.8, 0.5)  # m, forward x lateral, centred on the trunk
CRITIC_SCAN_RESOLUTION = 0.05


def _lift(cfg):
    for name in ("foot_clearance", "foot_swing_height"):
        cfg.rewards.pop(name, None)
    contact, heights = "feet_ground_contact", "foot_height_scan"
    common = dict(sensor_name=contact, height_sensor_name=heights, target_height=LIFT_TARGET,
                  command_name="twist", command_threshold=0.01)
    cfg.rewards["swing_foot_height"] = RewardTermCfg(
        func=microduck_mdp.swing_foot_height, weight=LIFT_WEIGHT,
        params=dict(common, max_air_time=LIFT_MAX_AIR_TIME))
    air = cfg.rewards["air_time"].params
    cfg.rewards["air_time"] = RewardTermCfg(
        func=microduck_mdp.feet_air_time_lifted, weight=AIR_TIME_WEIGHT,
        params=dict(common, threshold_min=air["threshold_min"], threshold_max=air["threshold_max"]))
    cfg.rewards["foot_stumble"] = RewardTermCfg(
        func=microduck_mdp.foot_stumble, weight=STUMBLE_WEIGHT, params={"sensor_name": contact})

    scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="trunk_base", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=CRITIC_SCAN_SIZE, resolution=CRITIC_SCAN_RESOLUTION),
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (scan,)
    cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
        func=microduck_mdp.height_scan_safe, params={"sensor_name": scan.name}, scale=1.0 / scan.max_distance)
    return cfg


def make_microduck_home_lift_env_cfg(play: bool = False):
    return _lift(make_microduck_home_robust_env_cfg(play=play))


# Speed with the lift walker. home-lift1 realises about 0.21 m/s at a 0.25 m/s command; the fetch
# gate asks for 0.3 m/s. Forward-only envs sample |vx| clamped to at least 0.3 m/s (mjlab's
# rel_forward_envs), so raising the top of the range to 0.5 and their share to 0.45 puts most of
# the forward experience at 0.3-0.5 m/s, where the wide speed term (std 0.35) still has gradient.
# Edges, lift and stumble terms are unchanged, so speed is bought only where it does not cost falls.
LIFT_FAST_LIN_VEL_X = (-0.2, 0.5)
LIFT_FAST_FORWARD_FRACTION = 0.45


def make_microduck_home_lift_fast_env_cfg(play: bool = False):
    cfg = make_microduck_home_lift_env_cfg(play=play)
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = LIFT_FAST_LIN_VEL_X
    twist.rel_forward_envs = LIFT_FAST_FORWARD_FRACTION
    return cfg



# Turning with the lift walker. home-lift1 realises 0.3 rad/s at a 1.0 rad/s command and 0.8 at
# 2.0; the fetch gate asks for 1 rad/s realised. Both yaw terms are flat at that shortfall (the
# tight one, std 0.4, pays 0.01% at 1.2 rad/s of error and the base one 6%), so turning harder
# earns nothing until the error is already small. A wide term (std 1.5) has gradient across the
# whole shortfall, as the wide speed term has for forward speed, and a larger turn-in-place share
# gives the samples.
LIFT_AGILE_WIDE_YAW_STD = 1.5
LIFT_AGILE_WIDE_YAW_WEIGHT = 2.0
LIFT_AGILE_TURN_IN_PLACE_FRACTION = 0.25


def make_microduck_home_lift_agile_env_cfg(play: bool = False):
    cfg = make_microduck_home_lift_fast_env_cfg(play=play)
    wide = deepcopy(cfg.rewards["track_angular_velocity"])
    wide.weight, wide.params["std"] = LIFT_AGILE_WIDE_YAW_WEIGHT, LIFT_AGILE_WIDE_YAW_STD
    cfg.rewards["track_angular_velocity_wide"] = wide
    cfg.commands["twist"].rel_turn_in_place_envs = LIFT_AGILE_TURN_IN_PLACE_FRACTION
    return cfg


# The yaw terms above act on the instantaneous yaw rate, which the waddle swings by about
# +-1.1 rad/s every step (see mdp.track_yaw_rate_averaged): home-agile1's yaw rewards barely moved
# in 2000 iterations. This variant adds tracking of the stride-averaged yaw rate, with a std
# wide enough to have gradient at the ~1 rad/s shortfall the walker starts from.
LIFT_TURN_AVG_YAW_STD = 1.0
LIFT_TURN_AVG_YAW_WEIGHT = 3.0
LIFT_TURN_AVG_YAW_TAU = 0.4  # s, about one stride


def make_microduck_home_lift_turn_env_cfg(play: bool = False):
    cfg = make_microduck_home_lift_agile_env_cfg(play=play)
    cfg.rewards["track_yaw_rate_averaged"] = RewardTermCfg(
        func=microduck_mdp.track_yaw_rate_averaged, weight=LIFT_TURN_AVG_YAW_WEIGHT,
        params={"std": LIFT_TURN_AVG_YAW_STD, "command_name": "twist", "tau": LIFT_TURN_AVG_YAW_TAU})
    return cfg


# Posture for the lift walker. The lift walkers stand and walk in a splayed crouch: after a stop
# home-agile1 holds both hip-yaw joints near their limit, one foot 6 cm ahead of the trunk and
# the other 6 cm behind, the trunk at 8.4 cm above the feet (home-robust1: feet side by side,
# 11.6 cm), and the pose reward sits at a quarter of home-robust1's. That moves the zone where the
# crouch-and-bite succeeds by 4 cm and makes 4% of crouches fall, and a held crouch loads the
# servos. The Gaussian pose term cannot pull the joints back from that far, so this variant adds
# linear costs: on all leg joints while standing, on hip yaw and roll at all times (the average
# over a stride should stay near zero even when turning), and on a trunk lower than 10 cm.
STANCE_STAND_WEIGHT = -4.0
STANCE_HIP_WEIGHT = -1.5
STANCE_HEIGHT_WEIGHT = -30.0
STANCE_MIN_HEIGHT = 0.10
_LEG_JOINTS = (r"^(?!passive_|.*neck.*|.*head.*).*",)
_HIP_SPLAY_JOINTS = (r".*hip_yaw.*", r".*hip_roll.*")


def make_microduck_home_lift_stance_env_cfg(play: bool = False):
    cfg = make_microduck_home_lift_turn_env_cfg(play=play)
    cfg.rewards["stand_posture"] = RewardTermCfg(
        func=microduck_mdp.leg_posture_l1, weight=STANCE_STAND_WEIGHT,
        params={"command_name": "twist", "standing_only": True,
                "asset_cfg": SceneEntityCfg("robot", joint_names=_LEG_JOINTS)})
    cfg.rewards["hip_splay"] = RewardTermCfg(
        func=microduck_mdp.leg_posture_l1, weight=STANCE_HIP_WEIGHT,
        params={"command_name": "twist", "standing_only": False,
                "asset_cfg": SceneEntityCfg("robot", joint_names=_HIP_SPLAY_JOINTS)})
    cfg.rewards["trunk_low"] = RewardTermCfg(
        func=microduck_mdp.trunk_below_height, weight=STANCE_HEIGHT_WEIGHT,
        params={"min_height": STANCE_MIN_HEIGHT})
    return cfg
