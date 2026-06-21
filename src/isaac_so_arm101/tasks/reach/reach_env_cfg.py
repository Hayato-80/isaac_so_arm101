# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils

# import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaac_so_arm101.tasks.reach.mdp import sim2real_observations
from isaac_so_arm101.tasks.reach.mdp import sim2real_randomization
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # pos_x=(-0.1, 0.3),
            # pos_y=(-0.2, 0.2),
            # pos_z=(0.1, 0.3),
            pos_x=(-0.1, 0.1),
            pos_y=(-0.25, -0.1),
            pos_z=(0.1, 0.3),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class Sim2RealPolicyCfg(ObsGroup):
        """
        Observations for sim2real policy (17D).
        
        Based on techniques from: https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41
        
        This reduced observation set focuses on task-relevant information:
        - joint_pos_normalized (6D)
        - target_delta (3D) - goal error  
        - wrist_state (1D)
        - jaw_state (1D)
        - previous_action (6D)
        
        Total: 17 dimensions
        
        Key benefits:
        - Avoids noisy joint velocity signals that don't transfer to real robot
        - Uses goal error instead of absolute EE position for fixed target tasks
        - More robust to sensor variations and real-world noise
        """
        # Normalized joint positions (6D) - scaled to [-1, 1]
        joint_pos_normalized = ObsTerm(
            func=sim2real_observations.joint_pos_normalized,
        )
        
        # Goal reach error (3D) - delta from current EE to target
        goal_reach_error = ObsTerm(
            func=sim2real_observations.goal_reach_error,
            params={
                "command_name": "ee_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_link"]),
            },
        )
        
        # Wrist roll state (1D)
        wrist_state = ObsTerm(func=sim2real_observations.wrist_state)
        
        # Jaw/gripper state (1D)
        jaw_state = ObsTerm(func=sim2real_observations.jaw_state)
        
        # Previous action (6D) - for policy stability
        actions = ObsTerm(func=sim2real_observations.previous_action)

        def __post_init__(self):
            self.enable_corruption = False  # No corruption for sim2real
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
    # Optional: Add sim2real policy as a separate group for comparison
    # Can be switched via configuration


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    randomize_robot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_action_pink_noise = EventTerm(
        func=sim2real_randomization.reset_pink_noise_state,
        mode="reset",
    )
    reset_action_pink_noise.enable = False


    action_gaussian_noise = EventTerm(
        func=sim2real_randomization.randomize_action_noise,
        mode="before_step",
        params={
            "std": 0.03,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    action_gaussian_noise.enable = False

    action_pink_noise = EventTerm(
        func=sim2real_randomization.randomize_action_pink_noise,
        mode="before_step",
        params={
            "std": 0.03,
            "num_scales": 4,
            "alpha_min": 0.75,
            "alpha_max": 0.98,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    action_pink_noise.enable = False



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        import copy
        # Deepcopy config class attributes to prevent config leakage across tasks
        self.scene = copy.deepcopy(self.scene)
        self.observations = copy.deepcopy(self.observations)
        self.actions = copy.deepcopy(self.actions)
        self.commands = copy.deepcopy(self.commands)
        self.rewards = copy.deepcopy(self.rewards)
        self.terminations = copy.deepcopy(self.terminations)
        self.events = copy.deepcopy(self.events)
        self.curriculum = copy.deepcopy(self.curriculum)

        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
