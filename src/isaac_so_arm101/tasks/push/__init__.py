import gymnasium as gym
from . import agents

# Base environments
gym.register(
    id="Isaac-SO-ARM101-Push-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PushCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Push-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PushCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

# Pink noise comparison environments
gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-No-Noise-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pink_noise_comparison_env_cfg:SoArm101PushSim2RealNoNoiseCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-Action-Noise-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pink_noise_comparison_env_cfg:SoArm101PushSim2RealActionNoiseCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-Obs-Noise-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pink_noise_comparison_env_cfg:SoArm101PushSim2RealObsNoiseCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

# Play variants
gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-No-Noise-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pink_noise_comparison_env_cfg:SoArm101PushSim2RealNoNoiseCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-Action-Noise-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pink_noise_comparison_env_cfg:SoArm101PushSim2RealActionNoiseCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-Obs-Noise-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pink_noise_comparison_env_cfg:SoArm101PushSim2RealObsNoiseCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

# Colored noise (randomized beta) environments
gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-Obs-Colored-Noise-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pink_noise_comparison_env_cfg:SoArm101PushSim2RealObsColoredNoiseCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-Both-Colored-Noise-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pink_noise_comparison_env_cfg:SoArm101PushSim2RealBothColoredNoiseCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-Both-Colored-Noise-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_pink_noise_comparison_env_cfg:SoArm101PushSim2RealBothColoredNoiseCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

# Fixed Beta Colored Noise Scenarios on Observations Only
for beta in ["025", "050", "075", "100", "125", "150"]:
    gym.register(
        id=f"Isaac-SO-ARM101-Push-Sim2Real-Obs-Beta{beta}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.sim2real_pink_noise_comparison_env_cfg:SoArm101PushSim2RealObsBeta{beta}Cfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
        },
        disable_env_checker=True,
    )
    
    gym.register(
        id=f"Isaac-SO-ARM101-Push-Sim2Real-Obs-Beta{beta}-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.sim2real_pink_noise_comparison_env_cfg:SoArm101PushSim2RealObsBeta{beta}Cfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
        },
        disable_env_checker=True,
    )

# Object Colored Noise Environments
gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-Obj-Colored-Noise-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_object_noise_env_cfg:SoArm101PushSim2RealObjColoredNoiseCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-Obj-Colored-Noise-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_object_noise_env_cfg:SoArm101PushSim2RealObjColoredNoiseCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-Obj-Both-Colored-Noise-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_object_noise_env_cfg:SoArm101PushSim2RealObjBothColoredNoiseCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Push-Sim2Real-Obj-Both-Colored-Noise-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim2real_object_noise_env_cfg:SoArm101PushSim2RealObjBothColoredNoiseCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

for beta in ["050", "075", "100", "125", "150"]:
    gym.register(
        id=f"Isaac-SO-ARM101-Push-Sim2Real-Obj-Beta{beta}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.sim2real_object_noise_env_cfg:SoArm101PushSim2RealObjBeta{beta}Cfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
        },
        disable_env_checker=True,
    )
    
    gym.register(
        id=f"Isaac-SO-ARM101-Push-Sim2Real-Obj-Beta{beta}-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.sim2real_object_noise_env_cfg:SoArm101PushSim2RealObjBeta{beta}Cfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
        },
        disable_env_checker=True,
    )
