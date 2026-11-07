"""Hinged lower jaw for the microduck, the real robot's 15th servo (Dynamixel ID 34).

The exported model fuses the jaw into the head link. The real beak is the orange plate along
the underside of the head: it hinges next to the servo at the rear of the head and its tip
swings down to open. Range and rest pose follow pollen-robotics/microduck
(`duck-control/src/model.rs`): closed -5 deg, fully open +30 deg.

The joint is named ``passive_mouth`` so every existing ``^(?!passive_).*`` selector keeps it
out of the 14-servo policy interface; the mouth is commanded separately, never by a policy.
"""

import math

import mujoco

HEAD_BODY = "jaw_soft"  # the head link carrying head_roll (the name predates this split)
JAW_MESHES = ("jaw", "jaw_soft", "soft_mouth_top")
JOINT_NAME = "passive_mouth"
# Hinge on the head link's lateral (y) axis, at the servo output; x/z from the xl330 mount.
PIVOT = (0.003, 0.0, -0.018)
MOUTH_CLOSED = -0.0121  # rad, the robot's rest value
MOUTH_OPEN = math.radians(30.0)
RANGE = (math.radians(-5.0), MOUTH_OPEN)
JAW_MASS = 0.02  # kg, moved out of the head link so the total mass is unchanged

_GEOM_FIELDS = ("name", "type", "meshname", "pos", "quat", "material", "group", "contype",
                "conaffinity", "condim", "friction", "rgba", "size", "priority", "solref", "solimp")


def add_mouth(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    head = spec.body(HEAD_BODY)
    jaw = head.add_body(name="jaw", pos=list(PIVOT))
    jaw.add_joint(name=JOINT_NAME, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0],
                  range=list(RANGE), limited=mujoco.mjtLimited.mjLIMITED_TRUE)
    jaw.mass = JAW_MASS
    jaw.inertia = [2e-5, 3e-5, 1e-5]
    jaw.ipos = [-PIVOT[0], 0.0, -0.04 - PIVOT[2]]  # plate centre: ~4 cm toward the tip
    head.mass -= JAW_MASS

    moved = []
    for geom in list(head.geoms):
        if geom.meshname in JAW_MESHES:
            attrs = {f: getattr(geom, f) for f in _GEOM_FIELDS if hasattr(geom, f)}
            attrs["classname"] = geom.classname
            moved.append(attrs)
            spec.delete(geom)
    for attrs in moved:
        # Geom poses were given in the head frame; the jaw frame is offset by PIVOT.
        attrs["pos"] = [p - o for p, o in zip(attrs["pos"], PIVOT)]
        cls = attrs.pop("classname")
        g = jaw.add_geom(**{k: v for k, v in attrs.items() if v is not None})
        if cls:
            g.classname = cls
    jaw.add_site(name="jaw_tip", pos=[-0.00809334 - PIVOT[0], 0.0, -0.0777383 - PIVOT[2]], group=3)
    return spec
