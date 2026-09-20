# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Reach-target world: FlyGym's flat ground plus a target marker and cylinder obstacles.

Adapter over `flygym.compose.FlatGroundWorld`; the simulator itself is not modified.

Two FlyGym design points shape this file:

- Every geom in FlyGym is created with ``contype=0, conaffinity=0`` and collisions exist
  only as explicit contact pairs (`_GroundContactMixin._set_ground_contact`). A plain
  obstacle geom would therefore be walked straight through. We extend
  `_set_ground_contact` so that, whenever FlyGym pairs the fly with the ground, we also
  pair the chosen fly body segments with every obstacle.
- Target and obstacles are mocap bodies, so their positions can be changed per episode
  through ``mj_data.mocap_pos`` without recompiling the model.
"""

from __future__ import annotations

import numpy as np

from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.compose.fly import BaseFly
from flygym.compose.physics import ContactParams
from flygym.utils.mjcf import CAMERA_MODES, GEOM_TYPES

# Where unused objects are parked (mm). Far outside any arena we sample from.
PARKED_XY = np.array([500.0, 500.0])


class ReachTargetWorld(FlatGroundWorld):
    """Flat ground with one target marker and ``n_obstacles`` vertical cylinders.

    Args:
        n_obstacles: Number of obstacle slots compiled into the model. Episodes may use
            fewer; unused ones are parked far away.
        obstacle_radius: Cylinder radius in mm.
        obstacle_height: Cylinder height in mm.
        target_radius: Radius of the (visual-only) target marker in mm.
        obstacle_contact_bodies: Fly body segments that physically collide with
            obstacles.
        topdown_camera_height: Height of the fixed top-down camera in mm.
    """

    def __init__(
        self,
        *,
        n_obstacles: int = 0,
        obstacle_radius: float = 1.0,
        obstacle_height: float = 2.0,
        target_radius: float = 0.75,
        obstacle_contact_bodies: ContactBodiesPreset = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
        topdown_camera_height: float = 45.0,
        name: str = "reach_target_world",
    ) -> None:
        super().__init__(name=name)
        self.obstacle_radius = obstacle_radius
        self.target_radius = target_radius
        self._obstacle_contact_bodies = ContactBodiesPreset(obstacle_contact_bodies)
        worldbody = self.mjcf_root.worldbody

        # Visual-only target: a short bright cylinder (no contact pairs -> no collision).
        self.target_body = worldbody.add_body(
            name="target", mocap=True, pos=(*PARKED_XY, 0.0)
        )
        self.target_geom = self.target_body.add_geom(
            name="target_geom",
            type=GEOM_TYPES["cylinder"],
            size=(target_radius, 0.5, 0),
            pos=(0, 0, 0.5),
            rgba=(0.9, 0.15, 0.15, 1.0),
            contype=0,
            conaffinity=0,
        )

        self.obstacle_bodies = []
        self.obstacle_geoms = []
        for i in range(n_obstacles):
            body = worldbody.add_body(
                name=f"obstacle_{i}", mocap=True, pos=(*PARKED_XY, 0.0)
            )
            geom = body.add_geom(
                name=f"obstacle_{i}_geom",
                type=GEOM_TYPES["cylinder"],
                size=(obstacle_radius, obstacle_height / 2, 0),
                pos=(0, 0, obstacle_height / 2),
                rgba=(0.2, 0.35, 0.8, 1.0),
                contype=0,
                conaffinity=0,
            )
            self.obstacle_bodies.append(body)
            self.obstacle_geoms.append(geom)

        # Fixed top-down camera over the arena origin (looks along -z).
        self.topdown_camera = worldbody.add_camera(
            name="topdown",
            mode=CAMERA_MODES["fixed"],
            pos=(0, 0, topdown_camera_height),
            xyaxes=(1, 0, 0, 0, 1, 0),
            fovy=45.0,
        )

        # Filled in by `_set_ground_contact`: names of fly geoms paired with obstacles.
        self.fly_obstacle_contact_geom_names: dict[str, list[str]] = {}

    def _set_ground_contact(
        self,
        fly: BaseFly,
        bodysegs_with_ground_contact: list[BodySegment],
        ground_contact_params: ContactParams,
    ) -> None:
        super()._set_ground_contact(
            fly, bodysegs_with_ground_contact, ground_contact_params
        )
        preset_cls = type(fly).CONTACT_BODIES_PRESET_CLASS
        segments = preset_cls(self._obstacle_contact_bodies.value).to_body_segments_list()
        names = []
        for segment in segments:
            for body_geom in fly.bodyseg_to_mjcfgeom[segment]:
                names.append(body_geom.name)
                for obstacle_geom in self.obstacle_geoms:
                    self.mjcf_root.add_pair(
                        geomname1=body_geom.name,
                        geomname2=obstacle_geom.name,
                        name=f"{body_geom.name}-{obstacle_geom.name}-obstacle",
                        friction=ground_contact_params.get_friction_tuple(),
                        solref=ground_contact_params.get_solref_tuple(),
                        solimp=ground_contact_params.get_solimp_tuple(),
                        margin=ground_contact_params.margin,
                    )
        self.fly_obstacle_contact_geom_names[fly.name] = names
