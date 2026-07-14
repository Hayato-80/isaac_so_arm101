from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaac_so_arm101.tasks.push.mdp as mdp
import isaac_so_arm101.tasks.reach.mdp.sim2real_randomization as sim2real_randomization
from .sim2real_env_cfg import SoArm101PushSim2RealBaseCfg

@configclass
class SoArm101PushPinkNoiseBaseCfg(SoArm101PushSim2RealBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # Action pink noise setup
        self.events.reset_action_pink_noise = EventTerm(
            func=sim2real_randomization.reset_pink_noise_state,
            mode="reset",
        )
        self.events.action_pink_noise = EventTerm(
            func=sim2real_randomization.randomize_action_pink_noise,
            mode="before_step",
            params={
                "std": 0.05,
                "num_scales": 4,
                "alpha_min": 0.75,
                "alpha_max": 0.98,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Base noise configs
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=sim2real_randomization.PinkNoiseObservationModelCfg(
                std=0.02,
                num_scales=4,
                alpha_min=0.35,
                alpha_max=0.5,
            ),
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=sim2real_randomization.PinkNoiseObservationModelCfg(
                std=0.02,
                num_scales=4,
                alpha_min=0.35,
                alpha_max=0.5,
            ),
        )

@configclass
class SoArm101PushSim2RealNoNoiseCfg(SoArm101PushPinkNoiseBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False
        self.events.action_pink_noise.enable = False
        self.events.reset_action_pink_noise.enable = False

@configclass
class SoArm101PushSim2RealNoNoiseCfg_PLAY(SoArm101PushSim2RealNoNoiseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5

@configclass
class SoArm101PushSim2RealActionNoiseCfg(SoArm101PushPinkNoiseBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_action_pink_noise.enable = True
        self.events.action_pink_noise.enable = True
        self.observations.policy.enable_corruption = False

@configclass
class SoArm101PushSim2RealActionNoiseCfg_PLAY(SoArm101PushSim2RealActionNoiseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.events.action_pink_noise.enable = False
        self.events.reset_action_pink_noise.enable = False

@configclass
class SoArm101PushSim2RealObsNoiseCfg(SoArm101PushPinkNoiseBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = True
        self.events.reset_action_pink_noise.enable = False
        self.events.action_pink_noise.enable = False

@configclass
class SoArm101PushSim2RealObsNoiseCfg_PLAY(SoArm101PushSim2RealObsNoiseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101PushSim2RealBothNoiseCfg(SoArm101PushPinkNoiseBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_action_pink_noise.enable = True
        self.events.action_pink_noise.enable = True
        self.observations.policy.enable_corruption = True


@configclass
class SoArm101PushSim2RealBothNoiseCfg_PLAY(SoArm101PushSim2RealBothNoiseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.action_pink_noise.enable = False
        self.events.reset_action_pink_noise.enable = False


@configclass
class SoArm101PushSim2RealObsColoredNoiseCfg(SoArm101PushPinkNoiseBaseCfg):
    def __post_init__(self):
        super().__post_init__()
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


@configclass
class SoArm101PushSim2RealBothColoredNoiseCfg(SoArm101PushPinkNoiseBaseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_action_pink_noise.enable = True
        self.events.action_pink_noise.enable = True
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

@configclass
class SoArm101PushSim2RealBothColoredNoiseCfg_PLAY(SoArm101PushSim2RealBothColoredNoiseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.action_pink_noise.enable = False
        self.events.reset_action_pink_noise.enable = False

# Fixed Beta Scenarios
def create_fixed_beta_cfg(beta: float):
    @configclass
    class FixedBetaCfg(SoArm101PushPinkNoiseBaseCfg):
        def __post_init__(self):
            super().__post_init__()
            self.observations.policy.enable_corruption = True
            colored_cfg = sim2real_randomization.ColoredNoiseObservationModelCfg(
                std=0.02,
                beta_min=beta,
                beta_max=beta,
            )
            self.observations.policy.joint_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                noise=colored_cfg,
            )
            self.observations.policy.joint_vel = ObsTerm(
                func=mdp.joint_vel_rel,
                noise=sim2real_randomization.ColoredNoiseObservationModelCfg(
                    std=0.02,
                    beta_min=beta,
                    beta_max=beta,
                ),
            )
    return FixedBetaCfg

SoArm101PushSim2RealObsBeta025Cfg = create_fixed_beta_cfg(0.25)
SoArm101PushSim2RealObsBeta050Cfg = create_fixed_beta_cfg(0.50)
SoArm101PushSim2RealObsBeta075Cfg = create_fixed_beta_cfg(0.75)
SoArm101PushSim2RealObsBeta100Cfg = create_fixed_beta_cfg(1.00)
SoArm101PushSim2RealObsBeta125Cfg = create_fixed_beta_cfg(1.25)
SoArm101PushSim2RealObsBeta150Cfg = create_fixed_beta_cfg(1.50)

# PLAY variants for fixed Beta scenarios
def create_fixed_beta_play_cfg(base_cfg):
    @configclass
    class FixedBetaPlayCfg(base_cfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 10
            self.scene.env_spacing = 2.5
            self.observations.policy.enable_corruption = False
    return FixedBetaPlayCfg

SoArm101PushSim2RealObsBeta025Cfg_PLAY = create_fixed_beta_play_cfg(SoArm101PushSim2RealObsBeta025Cfg)
SoArm101PushSim2RealObsBeta050Cfg_PLAY = create_fixed_beta_play_cfg(SoArm101PushSim2RealObsBeta050Cfg)
SoArm101PushSim2RealObsBeta075Cfg_PLAY = create_fixed_beta_play_cfg(SoArm101PushSim2RealObsBeta075Cfg)
SoArm101PushSim2RealObsBeta100Cfg_PLAY = create_fixed_beta_play_cfg(SoArm101PushSim2RealObsBeta100Cfg)
SoArm101PushSim2RealObsBeta125Cfg_PLAY = create_fixed_beta_play_cfg(SoArm101PushSim2RealObsBeta125Cfg)
SoArm101PushSim2RealObsBeta150Cfg_PLAY = create_fixed_beta_play_cfg(SoArm101PushSim2RealObsBeta150Cfg)
