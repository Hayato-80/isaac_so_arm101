# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MDP functions for sim2real domain randomization.
Based on techniques from: https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41

This module provides functions to apply various forms of domain randomization
to make policies trained in simulation robust for real-world deployment.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_random_robot_mass(
    env: "ManagerBasedRLEnv", 
    min_scale: float = 0.7, 
    max_scale: float = 1.3,
) -> torch.Tensor:
    """Sample random mass scaling factors for robot bodies."""
    num_envs = env.num_envs
    return torch.rand(num_envs, device=env.device) * (max_scale - min_scale) + min_scale


def get_random_object_mass(
    env: "ManagerBasedRLEnv", 
    min_scale: float = 0.5, 
    max_scale: float = 1.5,
) -> torch.Tensor:
    """Sample random mass scaling factors for objects."""
    num_envs = env.num_envs
    return torch.rand(num_envs, device=env.device) * (max_scale - min_scale) + min_scale


def get_random_friction(
    env: "ManagerBasedRLEnv", 
    min_value: float = 0.1, 
    max_value: float = 1.5,
) -> torch.Tensor:
    """Sample random friction coefficients using log-uniform distribution."""
    num_envs = env.num_envs
    log_min = torch.log(torch.tensor(min_value, device=env.device))
    log_max = torch.log(torch.tensor(max_value, device=env.device))
    return torch.exp(torch.rand(num_envs, device=env.device) * (log_max - log_min) + log_min)


def get_random_contact_offset(
    env: "ManagerBasedRLEnv", 
    min_offset: float = 0.0, 
    max_offset: float = 0.02,
) -> torch.Tensor:
    """Sample random contact offset perturbations."""
    num_envs = env.num_envs
    return torch.rand(num_envs, device=env.device) * (max_offset - min_offset) + min_offset


def get_random_joint_damping_scale(
    env: "ManagerBasedRLEnv", 
    joint_names: list[str],
    min_scale: float = 0.5, 
    max_scale: float = 1.5,
) -> torch.Tensor:
    """Sample random damping scaling factors for specific joints."""
    num_envs = env.num_envs
    return torch.rand(num_envs, device=env.device) * (max_scale - min_scale) + min_scale


def get_random_joint_stiffness_scale(
    env: "ManagerBasedRLEnv", 
    joint_names: list[str],
    min_scale: float = 0.5, 
    max_scale: float = 1.5,
) -> torch.Tensor:
    """Sample random stiffness scaling factors for specific joints."""
    num_envs = env.num_envs
    return torch.rand(num_envs, device=env.device) * (max_scale - min_scale) + min_scale


def get_random_actuation_noise(
    env: "ManagerBasedRLEnv", 
    std: float = 0.1,
) -> torch.Tensor:
    """Sample random actuation noise values for motor commands."""
    num_envs = env.num_envs
    return torch.randn(num_envs, device=env.device) * std


def get_random_base_pose_noise(
    env: "ManagerBasedRLEnv", 
    pos_std: float = 0.01,
    rot_std: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample random base pose perturbations."""
    num_envs = env.num_envs
    
    # Position noise (x, y, z)
    pos_noise = torch.randn(num_envs, 3, device=env.device) * pos_std
    
    # Rotation noise in Euler angles
    roll_noise = torch.randn(num_envs, device=env.device) * rot_std
    pitch_noise = torch.randn(num_envs, device=env.device) * rot_std  
    yaw_noise = torch.randn(num_envs, device=env.device) * rot_std
    
    return pos_noise, (roll_noise, pitch_noise, yaw_noise)


def get_random_table_friction(
    env: "ManagerBasedRLEnv", 
    min_value: float = 0.2, 
    max_value: float = 1.0,
) -> torch.Tensor:
    """Sample random table surface friction coefficients."""
    num_envs = env.num_envs
    log_min = torch.log(torch.tensor(min_value, device=env.device))
    log_max = torch.log(torch.tensor(max_value, device=env.device))
    return torch.exp(torch.rand(num_envs, device=env.device) * (log_max - log_min) + log_min)


def get_random_object_size(
    env: "ManagerBasedRLEnv", 
    min_scale: float = 0.85, 
    max_scale: float = 1.15,
) -> torch.Tensor:
    """Sample random object size scaling factors."""
    num_envs = env.num_envs
    return torch.rand(num_envs, device=env.device) * (max_scale - min_scale) + min_scale


def get_random_gravity(
    env: "ManagerBasedRLEnv", 
    min_g: float = 8.0, 
    max_g: float = 12.0,
) -> torch.Tensor:
    """Sample random gravity values (simulating different conditions)."""
    num_envs = env.num_envs
    return torch.rand(num_envs, device=env.device) * (max_g - min_g) + min_g