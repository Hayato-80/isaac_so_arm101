# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SO101 Reach Task - Sim2Real Domain Randomization Configuration

Based on techniques from: https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41

Key sim2real insights from the blog post:
1. Observation dimension reduction (from 26 to 17 dimensions)
   - Remove joint velocity (6D) - noisy in real robot
   - Remove EE absolute position (3D) - use target delta instead
   
2. Essential observations for sim2real transfer:
   - Normalized joint positions (6D)
   - Target delta / goal error (3D)  
   - Wrist state (1D)
   - Jaw state (1D)
   - Previous action (6D)

3. ROS 2 integration points for real robot deployment
"""

from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaac_so_arm101.tasks.reach.reach_env_cfg import ReachEnvCfg
from isaac_so_arm101.robots import SO_ARM100_CFG, SO_ARM101_CFG  # noqa: F401
from isaac_so_arm101.tasks.reach.mdp import sim2real_randomization

##
# Sim2Real environment configurations that extend ReachEnvCfg
##


@configclass
class SoArm101ReachSim2RealEnvCfg(ReachEnvCfg):
    """SO101 Reach Task - Sim2Real optimized environment with domain randomization."""

    def __post_init__(self):
        super().__post_init__()
        
        # Use SO101 robot instead of SO100
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # Switch to 17D sim2real observation format
        self.observations.policy = self.observations.Sim2RealPolicyCfg()
        
        # Configure arm action for joint position control
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.5,
            use_default_offset=True,
        )
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["gripper"],
        #     open_command_expr={"gripper": 0.0},
        #     close_command_expr={"gripper": 0.0},
        # )
        
        # Configure command generator for end-effector pose
        self.commands.ee_pose.body_name = "gripper_link"
        
        
        # Override rewards for SO101 gripper_link
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_link"]
        
        # 60 Hz simulation to match real robot frequency
        self.sim.dt = 1.0 / 60.0
        self.decimation = 1  # No decimation at 60 Hz

        # Pink-noise action perturbation for sim2real robustness
        self.events.reset_action_pink_noise.enable = True
        self.events.action_pink_noise.enable = True


@configclass
class SoArm101ReachSim2RealEnvCfg_HIGH(SoArm101ReachSim2RealEnvCfg):
    """SO101 Reach Task - High randomization for robust training."""

    def __post_init__(self):
        super().__post_init__()
        # More aggressive mass range (±40%)
        self.events.randomize_robot_mass.params["mass_distribution_params"] = (0.6, 1.4)
        self.events.randomize_robot_mass.params.pop("mass_range", None)


@configclass
class SoArm101ReachSim2RealEnvCfg_LOW(SoArm101ReachSim2RealEnvCfg):
    """SO101 Reach Task - Low randomization for fine-tuning."""

    def __post_init__(self):
        super().__post_init__()
        # Conservative mass range ±10%
        self.events.randomize_robot_mass.params["mass_distribution_params"] = (0.9, 1.1)
        self.events.randomize_robot_mass.params.pop("mass_range", None)


@configclass
class SoArm101ReachSim2RealEnvCfg_MEDIUM(SoArm101ReachSim2RealEnvCfg):
    """SO101 Reach Task - Medium randomization for standard sim2real training."""

    def __post_init__(self):
        super().__post_init__()
        # Standard mass range ±30% (recommended starting point)
        self.events.randomize_robot_mass.params["mass_distribution_params"] = (0.7, 1.3)
        self.events.randomize_robot_mass.params.pop("mass_range", None)


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
class SoArm101ReachSim2RealEnvCfg_PLAY(SoArm101ReachSim2RealEnvCfg):
    """SO101 Reach Task - Sim2Real for playing/testing (no randomization)."""

    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for testing
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Disable all domain randomization for clean evaluation
        self.events.randomize_robot_mass.enable = False