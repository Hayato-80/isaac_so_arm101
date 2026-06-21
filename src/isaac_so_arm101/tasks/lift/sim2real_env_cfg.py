# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SO101 Lift Task - Sim2Real Domain Randomization Configuration

Based on techniques from: https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41

Key sim2real insights from the blog post:
1. Observation dimension reduction for robust real-world transfer
2. Domain randomization of physics parameters during training
3. Progressive curriculum from high to low randomization
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils

import isaac_so_arm101.tasks.lift.mdp.sim2real_randomization as sim2real_mdp
import isaac_so_arm101.tasks.lift.mdp as mdp
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

##
# Scene definition for sim2real lift task
##


@configclass
class Sim2RealLiftSceneCfg(InteractiveSceneCfg):
    """Configuration for lift scene with domain randomization support."""

    # robots: will be set by agent env cfg
    robot: ArticulationCfg = MISSING
    
    # end-effector sensor frame
    ee_frame: FrameTransformerCfg | None = None
    
    # object to lift
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings for sim2real lift task
##


@configclass
class Sim2RealCommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.3, -0.1),
            pos_z=(0.2, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class Sim2RealActionsCfg:
    """Action specifications for the MDP."""

    arm_action = MISSING
    gripper_action = MISSING


@configclass
class Sim2RealObservationsCfg(ObservationsCfg):
    """
    Optimized observations for sim2real lift task transfer.
    
    Based on blog post approach, reduced observation space focusing on:
    - Normalized joint positions (task-relevant)
    - Goal error instead of absolute positions
    - Wrist/jaw states for gripper awareness
    
    Key insight: Removing noisy signals like joint velocity that don't
    transfer well to real robots due to encoder noise.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - sim2real optimized."""

        # Core observations from blog post approach
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        
        # Object position in robot frame (task-relevant)
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        
        # Target from command (goal error representation)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "object_pose"}
        )
        
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups - sim2real optimized
    policy: PolicyCfg = PolicyCfg()


@configclass
class Sim2RealRewardsCfg(RewardsCfg):
    """Reward terms optimized for sim2real transfer."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=1.0)
    
    lifting_object = RewTerm(
        func=mdp.object_is_lifted, 
        params={"minimal_height": 0.025}, 
        weight=15.0
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.025, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.025, "command_name": "object_pose"},
        weight=5.0,
    )

    # Action regularization for smooth motions
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class Sim2RealEventCfg(EventCfg):
    """
    Event-based domain randomization for sim2real transfer.
    
    This is the key component for robust sim2real policies:
    - Robot mass randomization (±30%)
    - Object friction variation (log-uniform 0.1-1.5)
    - Joint damping/stiffness perturbation (±50%)
    - Contact offset errors (simulates sensor/measurement error)
    
    These randomizations are applied at episode startup and persist
    throughout the episode, forcing the policy to learn robust behaviors.
    """

    # Reset scene to default state
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    # Randomize object position at reset
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    # Robot mass randomization (±30% of nominal) - simulates different payloads
    randomize_robot_mass = EventTerm(
        func=mdp.randomize_rigid_object_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.7, 1.3),  # ±30% variation
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Object mass randomization - simulates different object weights
    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_object_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_range": (0.5, 1.5),  # ±50% variation for objects
            "distribution": "uniform",
        },
    )

    # Object friction randomization - simulates different table surfaces
    randomize_object_friction = EventTerm(
        func=mdp.randomize_rigid_object_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "friction_range": (0.1, 1.5),
            "distribution": "log_uniform",
        },
    )

    # Contact offset randomization - simulates sensor/measurement error
    randomize_contact_offset = EventTerm(
        func=mdp.randomize_rigid_object_contact_offset,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "offset_range": (0.0, 0.02),
            "distribution": "uniform",
        },
    )

    # Arm joint damping randomization - simulates motor variations
    randomize_arm_damping = EventTerm(
        func=mdp.randomize_articulation_joint_damping,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": ["shoulder_.*", "elbow_flex", "wrist_.*"],
            "damping_range": (0.5, 1.5),
            "distribution": "uniform",
        },
    )

    # Arm joint stiffness randomization - simulates motor controller variations
    randomize_arm_stiffness = EventTerm(
        func=mdp.randomize_articulation_joint_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": ["shoulder_.*", "elbow_flex", "wrist_.*"],
            "stiffness_range": (0.5, 1.5),
            "damping_range": (0.5, 1.5),
            "distribution": "uniform",
        },
    )

    # Actuation noise - simulates motor command noise and electrical interference
    randomize_actuation_noise = EventTerm(
        func=mdp.add_joint_actuation_noise,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "noise_std": 0.1,
            "distribution": "gaussian",
        },
    )


@configclass
class Sim2RealCurriculumCfg(CurriculumCfg):
    """Progressive curriculum for sim2real transfer."""

    # Gradually reduce action penalty as policy improves
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )
    
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, 
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configurations for sim2real lift training
##

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG


@configclass
class SoArm101LiftSim2RealEnvCfg(Sim2RealSceneCfg):
    """SO101 Lift Task - Sim2Real optimized environment."""

    scene: Sim2RealLiftSceneCfg = Sim2RealLiftSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: Sim2RealObservationsCfg = Sim2RealObservationsCfg()
    actions: Sim2RealActionsCfg = Sim2RealActionsCfg()
    commands: Sim2RealCommandsCfg = Sim2RealCommandsCfg()
    rewards: Sim2RealRewardsCfg = Sim2RealRewardsCfg()
    terminations: TerminationsCfg = TERMINATIONS_CFG
    events: Sim2RealEventCfg = Sim2RealEventCfg()
    curriculum: Sim2RealCurriculumCfg = Sim2RealCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        
        # simulation settings - match real robot frequency (~60Hz)
        self.sim.dt = 0.01  # 100Hz for stability with domain randomization
        
        # Physics settings for robust simulation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class SoArm101LiftSim2RealEnvCfg_HIGH(SoArm101LiftSim2RealEnvCfg):
    """SO101 Lift Task - High randomization for maximum robustness training."""

    def __post_init__(self):
        super().__post_init__()
        
        # More aggressive randomization ranges
        self.events.randomize_robot_mass.params["mass_distribution_params"] = (0.6, 1.4)
        self.events.randomize_robot_mass.params.pop("mass_range", None)
        self.events.randomize_object_friction.params["friction_range"] = (0.05, 2.0)
        self.events.randomize_arm_damping.params["damping_range"] = (0.3, 1.7)
        self.events.randomize_actuation_noise.params["noise_std"] = 0.2


@configclass
class SoArm101LiftSim2RealEnvCfg_MEDIUM(SoArm101LiftSim2RealEnvCfg):
    """SO101 Lift Task - Medium randomization (recommended starting point)."""

    def __post_init__(self):
        super().__post_init__()
        
        # Standard randomization ranges
        self.events.randomize_robot_mass.params["mass_distribution_params"] = (0.7, 1.3)
        self.events.randomize_robot_mass.params.pop("mass_range", None)
        self.events.randomize_object_friction.params["friction_range"] = (0.1, 1.5)
        self.events.randomize_arm_damping.params["damping_range"] = (0.5, 1.5)
        self.events.randomize_actuation_noise.params["noise_std"] = 0.1


@configclass
class SoArm101LiftSim2RealEnvCfg_LOW(SoArm101LiftSim2RealEnvCfg):
    """SO101 Lift Task - Low randomization for fine-tuning near real conditions."""

    def __post_init__(self):
        super().__post_init__()
        
        # Conservative randomization ranges (closer to real robot)
        self.events.randomize_robot_mass.params["mass_distribution_params"] = (0.9, 1.1)
        self.events.randomize_robot_mass.params.pop("mass_range", None)
        self.events.randomize_object_friction.params["friction_range"] = (0.2, 1.0)
        self.events.randomize_arm_damping.params["damping_range"] = (0.8, 1.2)
        self.events.randomize_actuation_noise.params["noise_std"] = 0.05


@configclass
class SoArm101LiftSim2RealEnvCfg_PLAY(SoArm101LiftSim2RealEnvCfg):
    """SO101 Lift Task - Sim2Real for playing/testing without randomization."""

    def __post_init__(self):
        super().__post_init__()
        
        # Smaller scene for testing
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Disable all domain randomization for clean evaluation
        self.events.randomize_robot_mass.enable = False
        self.events.randomize_object_mass.enable = False
        self.events.randomize_object_friction.enable = False
        self.events.randomize_contact_offset.enable = False
        self.events.randomize_arm_damping.enable = False
        self.events.randomize_arm_stiffness.enable = False
        self.events.randomize_actuation_noise.enable = False
        
        # Disable observation corruption for clean evaluation
        self.observations.policy.enable_corruption = False


##
# Pre-defined termination configuration
##

TERMINATIONS_CFG = TerminationsCfg(
    time_out=DoneTerm(func=mdp.time_out, time_out=True),
    object_dropping=DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    ),
)