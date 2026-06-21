# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SO101 Reach Task - Sim2Real Domain Randomization Configuration

This file follows Isaac Lab's naming conventions for auto-registration.
Based on techniques from: https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41

Sim2real insights from the blog post:
1. Observation dimension reduction (from 26 to 17 dimensions)
   - Remove joint velocity (6D) - noisy in real robot
   - Remove EE absolute position (3D) - use target delta instead
   
2. Essential observations for sim2real transfer:
   - Normalized joint positions (6D)
   - Target delta / goal error (3D)  
   - Wrist state (1D)
   - Jaw state (1D)
   - Previous action (6D)

3. Domain randomization for robust policies:
   - Robot mass ±30% (simulates different payloads)
   - Friction variations (different table surfaces)
   - Joint damping/stiffness (motor/controller variations)
"""

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaac_so_arm101.robots import SO_ARM101_CFG  # noqa: F401
from isaac_so_arm101.tasks.reach.reach_env_cfg import ReachEnvCfg, ActionsCfg
from isaac_so_arm101.tasks.reach.mdp import sim2real_randomization

##
# Environment Configurations for Sim2Real Training
##


@configclass
class SoArm101ReachSim2RealEnvCfg(ReachEnvCfg):
    """
    SO101 Reach Task - Sim2Real optimized environment.
    
    This configuration is optimized for sim2real transfer with:
    - 17D observation space (optimized for real robot transfer)
    - Domain randomization on robot dynamics
    - 60 Hz control frequency matching real robot
    """

    def __post_init__(self):
        super().__post_init__()
        
        # Use SO101 robot
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Use sim2real observation format (17D)
        self.observations.policy = self.observations.Sim2RealPolicyCfg()
        
        # Configure arm action for 5 DOF (excluding gripper)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.5,
            use_default_offset=True,
        )
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["gripper"],
        #     open_command_expr={"gripper": 0.00},
        #     close_command_expr={"gripper": 0.0},
        # )
        
        # Override rewards for SO101 gripper_link
        self.rewards.end_effector_position_tracking.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["gripper_link"]
        )
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["gripper_link"]
        )
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["gripper_link"]
        )
        
        # Set command body name for SO101
        self.commands.ee_pose.body_name = "gripper_link"
        
        # Set 60 Hz control frequency to match real robot
        self.sim.dt = 1.0 / 60.0
        self.decimation = 1  # No decimation at 60 Hz

        # Pink-noise action perturbation for sim2real robustness
        self.events.reset_action_pink_noise.enable = True
        self.events.action_pink_noise.enable = True
        
        # Domain randomization for robust policies
        # Robot mass randomization (±30% - inherited from parent, this is the default)


@configclass
class SoArm101ReachSim2RealEnvCfg_HIGH(SoArm101ReachSim2RealEnvCfg):
    """
    SO101 Reach Task - High randomization for robust training.
    
    Use this for training policies that need to handle larger variations
    in robot dynamics and environmental conditions.
    """

    def __post_init__(self):
        super().__post_init__()
        
        # More aggressive mass range (±40%) - handles larger payload variations
        self.events.randomize_robot_mass.params["mass_range"] = (0.6, 1.4)


@configclass
class SoArm101ReachSim2RealEnvCfg_MEDIUM(SoArm101ReachSim2RealEnvCfg):
    """
    SO101 Reach Task - Medium randomization for standard sim2real training.
    
    This is the recommended starting point - balances between
    simulation speed and robustness to real-world variations.
    """

    def __post_init__(self):
        super().__post_init__()
        
        # Standard mass range (±30%) - default recommended range
        self.events.randomize_robot_mass.params["mass_range"] = (0.7, 1.3)


@configclass
class SoArm101ReachSim2RealEnvCfg_MEDIUM_GAUSSIAN(SoArm101ReachSim2RealEnvCfg_MEDIUM):
    """Medium randomization with Gaussian action noise for A/B comparison."""

    def __post_init__(self):
        super().__post_init__()
        self.events.action_pink_noise.enable = False
        self.events.reset_action_pink_noise.enable = False
        self.events.action_gaussian_noise.enable = True


@configclass
class SoArm101ReachSim2RealEnvCfg_MEDIUM_PINK(SoArm101ReachSim2RealEnvCfg_MEDIUM):
    """Medium randomization with pink action noise for A/B comparison."""

    def __post_init__(self):
        super().__post_init__()
        self.events.action_gaussian_noise.enable = False
        self.events.reset_action_pink_noise.enable = True
        self.events.action_pink_noise.enable = True


@configclass
class SoArm101ReachSim2RealEnvCfg_LOW(SoArm101ReachSim2RealEnvCfg):
    """
    SO101 Reach Task - Low randomization for fine-tuning.
    
    Use this for final fine-tuning when you already have a robust policy
    and want to optimize for the specific real robot setup.
    """

    def __post_init__(self):
        super().__post_init__()
        
        # Conservative mass range (±10%) - minimal variation
        self.events.randomize_robot_mass.params["mass_range"] = (0.9, 1.1)


@configclass
class SoArm101ReachSim2RealEnvCfg_PLAY(SoArm101ReachSim2RealEnvCfg):
    """
    SO101 Reach Task - Sim2Real for playing/inference (no randomization).
    
    Use this configuration for running trained policies in simulation
    to verify they work before deploying to the real robot.
    Smaller scene size for faster inference.
    """

    def __post_init__(self):
        super().__post_init__()
        
        # Reduce environment count for faster evaluation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Disable domain randomization for clean evaluation
        self.events.randomize_robot_mass.enable = False
        
        # Disable observation corruption
        self.observations.policy.enable_corruption = False
