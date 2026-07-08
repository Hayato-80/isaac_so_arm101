# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SO101 Reach Task - Pink Noise Comparison Environments

This configuration file defines environments for comparing the effects of pink noise
on sim2real transfer.

Four scenarios are defined:
1.  **No Pink Noise**: Baseline training without any pink noise.
2.  **Action Pink Noise**: Pink noise applied only to the actions.
3.  **Observation Pink Noise**: Pink noise applied only to the observations.
4.  **Action and Observation Pink Noise**: Pink noise applied to both actions and observations.
"""

from isaaclab.utils import configclass
from isaac_so_arm101.tasks.reach.joint_pos_env_cfg import SoArm101ReachEnvCfg
from isaac_so_arm101.robots import SO_ARM101_CFG
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg as ObsTerm
from isaac_so_arm101.tasks.reach.mdp import sim2real_randomization

##
# Base Configuration for Pink Noise Comparison
##


@configclass
class SoArm101ReachPinkNoiseBaseCfg(SoArm101ReachEnvCfg):
    """Base configuration for the pink noise comparison environments."""

    def __post_init__(self):
        super().__post_init__()
        # Use default policy observation format (25D)
        self.observations.policy = self.observations.PolicyCfg()

        # Enable default randomizations/noise
        self.observations.policy.enable_corruption = True
        self.events.randomize_robot_mass.enable = True
        # Disable all default action perturbations
        self.events.action_gaussian_noise.enable = False
        self.events.action_pink_noise.enable = False
        self.events.reset_action_pink_noise.enable = False

    def _make_obs_pink_noise_cfg(self):
        return sim2real_randomization.PinkNoiseObservationModelCfg(
            std=0.02,
            num_scales=4,
            alpha_min=0.35,
            alpha_max=0.5,
        )


##
# 1. No Pink Noise (Baseline)
##


@configclass
class SoArm101ReachSim2RealNoNoiseCfg(SoArm101ReachPinkNoiseBaseCfg):
    """Scenario 1: No pink noise applied."""

    def __post_init__(self):
        super().__post_init__()
        # All noise is disabled by default in the base class.
        pass


##
# 2. Pink Noise on Actions Only
##


@configclass
class SoArm101ReachSim2RealActionNoiseCfg(SoArm101ReachPinkNoiseBaseCfg):
    """Scenario 2: Pink noise applied to actions only."""

    def __post_init__(self):
        super().__post_init__()
        # Enable pink noise on actions
        self.events.reset_action_pink_noise.enable = True
        self.events.action_pink_noise.enable = True
        # Disable observation noise/corruption (action noise only)
        self.observations.policy.enable_corruption = False


##
# 3. Pink Noise on Observations Only
##


@configclass
class SoArm101ReachSim2RealObsNoiseCfg(SoArm101ReachPinkNoiseBaseCfg):
    """Scenario 3: Pink noise applied to observations only."""

    def __post_init__(self):
        super().__post_init__()
        # Enable observation corruption
        self.observations.policy.enable_corruption = True
        # Apply pink noise to observations (replacing Unoise)
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=self._make_obs_pink_noise_cfg()
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=self._make_obs_pink_noise_cfg()
        )


##
# 4. Pink Noise on Actions and Observations
# ##


@configclass
class SoArm101ReachSim2RealBothNoiseCfg(SoArm101ReachPinkNoiseBaseCfg):
    """Scenario 4: Pink noise on both actions and observations."""

    def __post_init__(self):
        super().__post_init__()
        # Enable pink noise on actions
        self.events.reset_action_pink_noise.enable = True
        self.events.action_pink_noise.enable = True
        # Enable observation corruption
        self.observations.policy.enable_corruption = True
        # Apply pink noise to observations (replacing Unoise)
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=self._make_obs_pink_noise_cfg()
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=self._make_obs_pink_noise_cfg()
        )


##
# PLAY variants for evaluation
##


@configclass
class SoArm101ReachSim2RealNoNoiseCfg_PLAY(SoArm101ReachSim2RealNoNoiseCfg):
    """Play variant for Scenario 1."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # Disable noise/corruption for clean evaluation
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101ReachSim2RealActionNoiseCfg_PLAY(SoArm101ReachSim2RealActionNoiseCfg):
    """Play variant for Scenario 2."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # Disable noise/corruption for clean evaluation
        self.observations.policy.enable_corruption = False
        self.events.action_pink_noise.enable = False
        self.events.reset_action_pink_noise.enable = False


@configclass
class SoArm101ReachSim2RealObsNoiseCfg_PLAY(SoArm101ReachSim2RealObsNoiseCfg):
    """Play variant for Scenario 3."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # Disable noise/corruption for clean evaluation
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101ReachSim2RealBothNoiseCfg_PLAY(SoArm101ReachSim2RealBothNoiseCfg):
    """Play variant for Scenario 4."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # Disable noise/corruption for clean evaluation
        self.observations.policy.enable_corruption = False
        self.events.action_pink_noise.enable = False
        self.events.reset_action_pink_noise.enable = False


##
# 5. Colored Noise on Observations Only (beta randomized per episode)
##


@configclass
class SoArm101ReachSim2RealObsColoredNoiseCfg(SoArm101ReachPinkNoiseBaseCfg):
    """Scenario 5: Colored noise (randomized beta) applied to observations only.

    Each environment samples beta ~ U(beta_min, beta_max) at every episode reset.
    This provides a richer distribution of noise spectra than fixed pink noise (beta=1),
    improving policy robustness to unknown real-world sensor characteristics.
    """

    def __post_init__(self):
        super().__post_init__()
        # Enable observation corruption
        self.observations.policy.enable_corruption = True
        # Apply colored noise with randomized beta to observations
        colored_cfg = sim2real_randomization.ColoredNoiseObservationModelCfg(
            std=0.02,
            beta_min=0.5,
            beta_max=1.5,
        )
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=colored_cfg,
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=sim2real_randomization.ColoredNoiseObservationModelCfg(
                std=0.02,
                beta_min=0.5,
                beta_max=1.5,
            ),
        )


##
# 6. Colored Noise on Both Actions and Observations
##


@configclass
class SoArm101ReachSim2RealBothColoredNoiseCfg(SoArm101ReachPinkNoiseBaseCfg):
    """Scenario 6: Colored noise (randomized beta) on both actions and observations."""

    def __post_init__(self):
        super().__post_init__()
        # Enable pink noise on actions (action noise keeps fixed beta=1)
        self.events.reset_action_pink_noise.enable = True
        self.events.action_pink_noise.enable = True
        # Enable colored observation noise with randomized beta
        self.observations.policy.enable_corruption = True
        colored_cfg = sim2real_randomization.ColoredNoiseObservationModelCfg(
            std=0.02,
            beta_min=0.5,
            beta_max=1.5,
        )
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=colored_cfg,
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=sim2real_randomization.ColoredNoiseObservationModelCfg(
                std=0.02,
                beta_min=0.5,
                beta_max=1.5,
            ),
        )


##
# PLAY variants for new colored noise scenarios
##


@configclass
class SoArm101ReachSim2RealObsColoredNoiseCfg_PLAY(SoArm101ReachSim2RealObsColoredNoiseCfg):
    """Play variant for Scenario 5."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101ReachSim2RealBothColoredNoiseCfg_PLAY(SoArm101ReachSim2RealBothColoredNoiseCfg):
    """Play variant for Scenario 6."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.action_pink_noise.enable = False
        self.events.reset_action_pink_noise.enable = False

