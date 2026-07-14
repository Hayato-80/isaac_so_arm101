from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaac_so_arm101.tasks.push.mdp as mdp
import isaac_so_arm101.tasks.reach.mdp.sim2real_randomization as sim2real_randomization
from .sim2real_pink_noise_comparison_env_cfg import SoArm101PushPinkNoiseBaseCfg

@configclass
class SoArm101PushSim2RealObjColoredNoiseCfg(SoArm101PushPinkNoiseBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True
        
        # Remove observation noise from joint_pos and joint_vel to isolate object noise
        self.observations.policy.joint_pos.noise = None
        self.observations.policy.joint_vel.noise = None
        
        colored_cfg = sim2real_randomization.ColoredNoiseObservationModelCfg(
            std=0.02,
            beta_min=0.5,
            beta_max=1.5,
        )
        self.observations.policy.object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            noise=colored_cfg,
        )
        self.events.reset_action_pink_noise.enable = False
        self.events.action_pink_noise.enable = False

@configclass
class SoArm101PushSim2RealObjBothColoredNoiseCfg(SoArm101PushPinkNoiseBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True
        
        # Remove observation noise from joint_pos and joint_vel to isolate object noise
        self.observations.policy.joint_pos.noise = None
        self.observations.policy.joint_vel.noise = None
        
        colored_cfg = sim2real_randomization.ColoredNoiseObservationModelCfg(
            std=0.02,
            beta_min=0.5,
            beta_max=1.5,
        )
        self.observations.policy.object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            noise=colored_cfg,
        )
        self.events.reset_action_pink_noise.enable = True
        self.events.action_pink_noise.enable = True

@configclass
class SoArm101PushSim2RealObjBothColoredNoiseCfg_PLAY(SoArm101PushSim2RealObjBothColoredNoiseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.action_pink_noise.enable = False
        self.events.reset_action_pink_noise.enable = False

# Fixed Beta Scenarios
def create_fixed_beta_obj_cfg(beta: float):
    @configclass
    class FixedBetaObjCfg(SoArm101PushPinkNoiseBaseCfg):
        def __post_init__(self):
            super().__post_init__()
            self.observations.policy.enable_corruption = True
            
            # Remove observation noise from joint_pos and joint_vel to isolate object noise
            self.observations.policy.joint_pos.noise = None
            self.observations.policy.joint_vel.noise = None
            
            colored_cfg = sim2real_randomization.ColoredNoiseObservationModelCfg(
                std=0.02,
                beta_min=beta,
                beta_max=beta,
            )
            self.observations.policy.object_position = ObsTerm(
                func=mdp.object_position_in_robot_root_frame,
                noise=colored_cfg,
            )
            self.events.reset_action_pink_noise.enable = False
            self.events.action_pink_noise.enable = False
    return FixedBetaObjCfg

SoArm101PushSim2RealObjBeta050Cfg = create_fixed_beta_obj_cfg(0.50)
SoArm101PushSim2RealObjBeta075Cfg = create_fixed_beta_obj_cfg(0.75)
SoArm101PushSim2RealObjBeta100Cfg = create_fixed_beta_obj_cfg(1.00)
SoArm101PushSim2RealObjBeta125Cfg = create_fixed_beta_obj_cfg(1.25)
SoArm101PushSim2RealObjBeta150Cfg = create_fixed_beta_obj_cfg(1.50)

# PLAY variants for fixed Beta scenarios
def create_fixed_beta_obj_play_cfg(base_cfg):
    @configclass
    class FixedBetaObjPlayCfg(base_cfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 10
            self.scene.env_spacing = 2.5
            self.observations.policy.enable_corruption = False
            self.events.action_pink_noise.enable = False
            self.events.reset_action_pink_noise.enable = False
    return FixedBetaObjPlayCfg

SoArm101PushSim2RealObjColoredNoiseCfg_PLAY = create_fixed_beta_obj_play_cfg(SoArm101PushSim2RealObjColoredNoiseCfg)
SoArm101PushSim2RealObjBeta050Cfg_PLAY = create_fixed_beta_obj_play_cfg(SoArm101PushSim2RealObjBeta050Cfg)
SoArm101PushSim2RealObjBeta075Cfg_PLAY = create_fixed_beta_obj_play_cfg(SoArm101PushSim2RealObjBeta075Cfg)
SoArm101PushSim2RealObjBeta100Cfg_PLAY = create_fixed_beta_obj_play_cfg(SoArm101PushSim2RealObjBeta100Cfg)
SoArm101PushSim2RealObjBeta125Cfg_PLAY = create_fixed_beta_obj_play_cfg(SoArm101PushSim2RealObjBeta125Cfg)
SoArm101PushSim2RealObjBeta150Cfg_PLAY = create_fixed_beta_obj_play_cfg(SoArm101PushSim2RealObjBeta150Cfg)
