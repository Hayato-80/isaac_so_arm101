from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
import isaac_so_arm101.tasks.reach.mdp.sim2real_randomization as sim2real_randomization
from .sim2real_pink_noise_comparison_env_cfg import SoArm101ReachPinkNoiseBaseCfg

@configclass
class SoArm101ReachSim2RealCmdColoredNoiseCfg(SoArm101ReachPinkNoiseBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True
        
        # Remove observation noise from joint_pos and joint_vel to isolate command noise
        self.observations.policy.joint_pos.noise = None
        self.observations.policy.joint_vel.noise = None
        
        colored_cfg = sim2real_randomization.ColoredNoiseObservationModelCfg(
            std=0.02,
            beta_min=0.5,
            beta_max=1.5,
        )
        self.observations.policy.pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"},
            noise=colored_cfg,
        )
        self.events.reset_action_pink_noise.enable = False
        self.events.action_pink_noise.enable = False

@configclass
class SoArm101ReachSim2RealCmdBothColoredNoiseCfg(SoArm101ReachPinkNoiseBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True
        
        # Remove observation noise from joint_pos and joint_vel to isolate command noise
        self.observations.policy.joint_pos.noise = None
        self.observations.policy.joint_vel.noise = None
        
        colored_cfg = sim2real_randomization.ColoredNoiseObservationModelCfg(
            std=0.02,
            beta_min=0.5,
            beta_max=1.5,
        )
        self.observations.policy.pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_pose"},
            noise=colored_cfg,
        )
        self.events.reset_action_pink_noise.enable = True
        self.events.action_pink_noise.enable = True

@configclass
class SoArm101ReachSim2RealCmdBothColoredNoiseCfg_PLAY(SoArm101ReachSim2RealCmdBothColoredNoiseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.action_pink_noise.enable = False
        self.events.reset_action_pink_noise.enable = False

# Fixed Beta Scenarios
def create_fixed_beta_cmd_cfg(beta: float):
    @configclass
    class FixedBetaCmdCfg(SoArm101ReachPinkNoiseBaseCfg):
        def __post_init__(self):
            super().__post_init__()
            self.observations.policy.enable_corruption = True
            
            # Remove observation noise from joint_pos and joint_vel to isolate command noise
            self.observations.policy.joint_pos.noise = None
            self.observations.policy.joint_vel.noise = None
            
            colored_cfg = sim2real_randomization.ColoredNoiseObservationModelCfg(
                std=0.02,
                beta_min=beta,
                beta_max=beta,
            )
            self.observations.policy.pose_command = ObsTerm(
                func=mdp.generated_commands,
                params={"command_name": "ee_pose"},
                noise=colored_cfg,
            )
            self.events.reset_action_pink_noise.enable = False
            self.events.action_pink_noise.enable = False
    return FixedBetaCmdCfg

SoArm101ReachSim2RealCmdBeta050Cfg = create_fixed_beta_cmd_cfg(0.50)
SoArm101ReachSim2RealCmdBeta075Cfg = create_fixed_beta_cmd_cfg(0.75)
SoArm101ReachSim2RealCmdBeta100Cfg = create_fixed_beta_cmd_cfg(1.00)
SoArm101ReachSim2RealCmdBeta125Cfg = create_fixed_beta_cmd_cfg(1.25)
SoArm101ReachSim2RealCmdBeta150Cfg = create_fixed_beta_cmd_cfg(1.50)

# PLAY variants for fixed Beta scenarios
def create_fixed_beta_cmd_play_cfg(base_cfg):
    @configclass
    class FixedBetaCmdPlayCfg(base_cfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 10
            self.scene.env_spacing = 2.5
            self.observations.policy.enable_corruption = False
            self.events.action_pink_noise.enable = False
            self.events.reset_action_pink_noise.enable = False
    return FixedBetaCmdPlayCfg

SoArm101ReachSim2RealCmdColoredNoiseCfg_PLAY = create_fixed_beta_cmd_play_cfg(SoArm101ReachSim2RealCmdColoredNoiseCfg)
SoArm101ReachSim2RealCmdBeta050Cfg_PLAY = create_fixed_beta_cmd_play_cfg(SoArm101ReachSim2RealCmdBeta050Cfg)
SoArm101ReachSim2RealCmdBeta075Cfg_PLAY = create_fixed_beta_cmd_play_cfg(SoArm101ReachSim2RealCmdBeta075Cfg)
SoArm101ReachSim2RealCmdBeta100Cfg_PLAY = create_fixed_beta_cmd_play_cfg(SoArm101ReachSim2RealCmdBeta100Cfg)
SoArm101ReachSim2RealCmdBeta125Cfg_PLAY = create_fixed_beta_cmd_play_cfg(SoArm101ReachSim2RealCmdBeta125Cfg)
SoArm101ReachSim2RealCmdBeta150Cfg_PLAY = create_fixed_beta_cmd_play_cfg(SoArm101ReachSim2RealCmdBeta150Cfg)
