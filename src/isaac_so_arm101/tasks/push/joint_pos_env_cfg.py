import isaaclab_tasks.manager_based.manipulation.lift.mdp as lift_mdp
import isaac_so_arm101.tasks.push.mdp as mdp
from isaaclab.assets import RigidObjectCfg

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
import isaaclab.sim as sim_utils

from isaaclab.utils import configclass
from isaac_so_arm101.robots import SO_ARM101_CFG  # We focus on SO_ARM101
from isaac_so_arm101.tasks.push.push_env_cfg import SoArm101PushEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class SoArm101PushCubeEnvCfg(SoArm101PushEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = lift_mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["gripper_link"]

        # Set 30mm blue Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.10, 0.0, 0.015], rot=[1, 0, 0, 0]),
            spawn=CuboidCfg(
                size=(0.03, 0.03, 0.03),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.05),
                collision_props=CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.01, 0.0, -0.09],
                    ),
                ),
            ],
        )


@configclass
class SoArm101PushCubeEnvCfg_PLAY(SoArm101PushCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
