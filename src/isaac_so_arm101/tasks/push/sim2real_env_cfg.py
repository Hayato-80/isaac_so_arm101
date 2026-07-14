from dataclasses import MISSING

import isaaclab.sim as sim_utils

import isaac_so_arm101.tasks.push.mdp as mdp
import isaac_so_arm101.tasks.reach.mdp.sim2real_randomization as sim2real_randomization
from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaac_so_arm101.tasks.push.push_env_cfg import ObservationsCfg, RewardsCfg, EventCfg
from isaac_so_arm101.tasks.push.joint_pos_env_cfg import SoArm101PushCubeEnvCfg

@configclass
class Sim2RealPushObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        target_object_position = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "object_pose"}
        )
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Sim2RealPushRewardsCfg(RewardsCfg):
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05, "object_cfg": SceneEntityCfg("object"), "ee_frame_cfg": SceneEntityCfg("ee_frame")}, weight=1.0)
    
    pushing_object = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.1, "command_name": "object_pose", "robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")},
        weight=2.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class Sim2RealPushEventCfg(EventCfg):
    # Action noise
    action_noise = EventTerm(
        func=sim2real_randomization.randomize_action_noise,
        mode="before_step",
        params={
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot")
        },
    )


@configclass
class SoArm101PushSim2RealBaseCfg(SoArm101PushCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.observations = Sim2RealPushObservationsCfg()
        self.rewards = Sim2RealPushRewardsCfg()
        
        self.events.action_noise = EventTerm(
            func=sim2real_randomization.randomize_action_noise,
            mode="before_step",
            params={
                "std": 0.05,
                "asset_cfg": SceneEntityCfg("robot")
            },
        )
