# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SO101 Sim2Real Observation Functions

Based on techniques from: https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41

Key insights from the blog post:
1. Original 26D observations were too complex and didn't transfer well to real robot
   - Joint velocity was noisy in real robot due to encoder noise
   - EE absolute position wasn't needed for fixed target tasks

2. Final 17D observation structure (recommended):
   - joint_pos_normalized: 6D (normalized joint positions)
   - target_delta: 3D (goal error / delta to target)
   - wrist_state: 1D (wrist_roll position)
   - jaw_state: 1D (gripper position)  
   - previous_action: 6D (last action for policy stability)

This approach focuses on what's actually needed for the task rather than everything available.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def _sanitize_joint_positions(joint_pos: torch.Tensor, joint_ranges: torch.Tensor) -> torch.Tensor:
    """Replace non-finite joint values and clamp to configured joint limits."""
    max_vals = joint_ranges[:, 0]
    min_vals = joint_ranges[:, 1]
    centers = 0.5 * (max_vals + min_vals)
    centers = centers.unsqueeze(0).expand_as(joint_pos)

    finite_joint_pos = torch.where(torch.isfinite(joint_pos), joint_pos, centers)
    return torch.clamp(finite_joint_pos, min=min_vals, max=max_vals)


def joint_pos_normalized(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    Get normalized joint positions.
    
    Normalization based on typical SO101 joint ranges:
    - shoulder_pan: [-2.88, 2.88] rad (~±165°)
    - shoulder_lift: [-2.69, 2.69] rad (~±154°)  
    - elbow_flex: [0, 3.05] rad (0-175°)
    - wrist_pitch: [-2.69, 2.69] rad (~±154°)
    - wrist_roll: [-3.05, 3.05] rad (~±175°)
    - jaw: [0, 0.04] m (gripper opening)
    
    Returns normalized values in range [-1, 1].
    """
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos[:, :6]  # Get only first 6 joints (exclude gripper if needed)
    
    # SO101 joint name order and normalization ranges [max, min]
    joint_ranges = torch.tensor([
        [2.88, -2.88],   # shoulder_pan: center at 0, range ±2.88
        [2.69, -2.69],   # shoulder_lift: center at 0, range ±2.69
        [3.05, 0.0],     # elbow_flex: range 0 to 3.05
        [2.69, -2.69],   # wrist_pitch: center at 0, range ±2.69  
        [3.05, -3.05],   # wrist_roll: center at 0, range ±3.05
        [0.04, 0.0],     # jaw: range 0 to 0.04m
    ], device=joint_pos.device, dtype=joint_pos.dtype)

    # Guard against corrupted simulator/bridge values before normalization.
    joint_pos = _sanitize_joint_positions(joint_pos, joint_ranges)
    
    # Normalize: (value - min) / (max - min) * 2 - 1 to get [-1, 1]
    # ranges[:, 0] is max, ranges[:, 1] is min
    max_vals = joint_ranges[:, 0]
    min_vals = joint_ranges[:, 1]
    normalized = 2.0 * (joint_pos - min_vals) / (max_vals - min_vals) - 1.0
    
    return normalized


def goal_reach_error(
    env: "ManagerBasedRLEnv", 
    command_name: str,
    asset_cfg: "SceneEntityCfg | None" = None,
) -> torch.Tensor:
    """
    Get the error between current EE position and target.
    
    This is the key observation from the blog post - using delta to target
    instead of absolute EE position makes the policy more robust for sim2real transfer.
    
    Args:
        env: The environment instance
        command_name: Name of the pose command containing the target
        asset_cfg: Scene entity configuration for robot
        
    Returns:
        3D vector representing (x, y, z) error from current to target position.
    """
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg
        asset_cfg = SceneEntityCfg("robot")
    
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Get EE body index for gripper/end-effector
    asset_cfg.resolve(env.scene)
    body_indices = asset_cfg.body_ids
    
    # Current EE position in world frame
    # body_state_w shape: (num_envs, num_bodies, 13) - pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
    ee_pos_w = robot.data.body_pos_w[:, body_indices[0], :3]
    
    # Target position from command (in robot base frame)
    target_pos_b = command[:, :3]
    
    # Transform target from base frame to world frame
    robot_base_pos_w = robot.data.root_pos_w[:, :3]
    robot_quat_w = robot.data.root_quat_w
    
    # Use Isaac Lab utility for frame transformation
    from isaaclab.utils.math import combine_frame_transforms
    target_pos_w, _ = combine_frame_transforms(robot_base_pos_w, robot_quat_w, target_pos_b)
    
    # Error vector (target - current)
    error = target_pos_w - ee_pos_w
    
    return error


def wrist_state(env: "ManagerBasedRLEnv", asset_cfg: "SceneEntityCfg | None" = None) -> torch.Tensor:
    """Get wrist_roll joint position."""
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg
        asset_cfg = SceneEntityCfg("robot")
    
    robot = env.scene[asset_cfg.name]
    joint_names = robot.data.joint_names
    
    # Find wrist_roll joint
    for i, name in enumerate(joint_names):
        if "wrist" in name.lower() and "roll" in name.lower():
            wrist = robot.data.joint_pos[:, i:i+1]
            wrist = torch.where(torch.isfinite(wrist), wrist, torch.zeros_like(wrist))
            return torch.clamp(wrist, min=-3.05, max=3.05)
    
    # Fallback: return zeros if not found
    num_envs = robot.data.joint_pos.shape[0]
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    return torch.zeros((num_envs, 1), device=device, dtype=dtype)



def jaw_state(env: "ManagerBasedRLEnv", asset_cfg: "SceneEntityCfg | None" = None) -> torch.Tensor:
    """Get gripper/jaw position for SO101."""
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg
        asset_cfg = SceneEntityCfg("robot")
    
    robot = env.scene[asset_cfg.name]
    joint_names = robot.data.joint_names
    
    # SO101 has 'jaw' or 'gripper' joint - find it specifically
    for i, name in enumerate(joint_names):
        if "jaw" in name.lower():
            jaw = robot.data.joint_pos[:, i:i+1]
            jaw = torch.where(torch.isfinite(jaw), jaw, torch.zeros_like(jaw))
            return torch.clamp(jaw, min=0.0, max=0.04)
    
    # Fallback: search for gripper
    for i, name in enumerate(joint_names):
        if "gripper" in name.lower():
            jaw = robot.data.joint_pos[:, i:i+1]
            jaw = torch.where(torch.isfinite(jaw), jaw, torch.zeros_like(jaw))
            return torch.clamp(jaw, min=0.0, max=0.04)
    
    # Final fallback: return zeros if not found
    num_envs = robot.data.joint_pos.shape[0]
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    return torch.zeros((num_envs, 1), device=device, dtype=dtype)



def previous_action(env: "ManagerBasedRLEnv", action_term_name: str = "arm_action") -> torch.Tensor:
    """
    Get the last action applied to the robot.
    
    Args:
        env: The environment instance
        action_term_name: Name of the action term to retrieve (default: "arm_action")
        
    Returns:
        Last action tensor of shape (num_envs, action_dim)
    """
    if hasattr(env, "action_manager") and env.action_manager is not None:
        try:
            # Try to get action from action manager
            action = env.action_manager.action
            if action is not None and hasattr(action, "shape"):
                return action
        except Exception:
            pass
    
    # Fallback: return zeros if action not available
    num_envs = env.num_envs
    device = env.device
    # Assume 6D action for SO101
    return torch.zeros((num_envs, 6), device=device, dtype=torch.float32)


def _rotate_point(point: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    """
    Rotate a point by a quaternion.
    
    Args:
        point: (N, 3) tensor of points in original frame
        quaternion: (N, 4) tensor of quaternions [x, y, z, w]
        
    Returns:
        Rotated points in new frame (N, 3)
    """
    x, y, z, w = quaternion.unbind(dim=-1)
    
    # Quaternion rotation formula
    ix = w * x + y * z
    iy = w * y - z * x
    iz = w * z + x * y
    iw = 1 - 2 * (x**2 + y**2)
    
    rx = point[:, 0] * (iw + 2*x**2 - 1) + point[:, 1] * (2*ix - 2*z*y) + point[:, 2] * (2*iy + 2*z*x)
    ry = point[:, 0] * (2*ix + 2*z*y) + point[:, 1] * (iw + 2*y**2 - 1) + point[:, 2] * (-2*iz + 2*x*w)  
    rz = point[:, 0] * (2*iy - 2*z*x) + point[:, 1] * (2*iz + 2*x*w) + point[:, 2] * (iw + 2*z**2 - 1)
    
    return torch.stack([rx, ry, rz], dim=-1)


def build_sim2real_observation(env: "ManagerBasedRLEnv", command_name: str = "ee_pose") -> torch.Tensor:
    """
    Build the complete 17D observation vector for sim2real transfer.
    
    This combines all essential observations as recommended in the blog post:
    - joint_pos_normalized (6D)
    - target_delta (3D)  
    - wrist_state (1D)
    - jaw_state (1D)
    - previous_action (6D)
    
    Total: 17 dimensions
    
    This reduced observation space focuses on task-relevant information and
    avoids noisy signals that don't transfer well to real robots.
    
    Args:
        env: The environment instance
        command_name: Name of the pose command containing the target
        
    Returns:
        Concatenated observation tensor of shape (num_envs, 17)
    """
    obs_list = [
        joint_pos_normalized(env),                          # 6D
        goal_reach_error(env, command_name),                # 3D
        wrist_state(env),                                   # 1D
        jaw_state(env),                                     # 1D  
        previous_action(env),                               # 6D
    ]
    
    return torch.cat(obs_list, dim=-1)


def build_full_observation(env: "ManagerBasedRLEnv", command_name: str = "ee_pose") -> torch.Tensor:
    """
    Build full observation vector (26D - original approach from blog post).
    
    This includes all available observations but was found to not transfer well.
    Kept here for reference and comparison.
    
    Structure:
    - joint_pos_normalized (6D)
    - joint_vel (6D) - noisy in real robot!
    - ee_pos_w (3D) - absolute position, not needed for fixed target
    - target_delta (3D)
    - wrist_state (1D)
    - jaw_state (1D)
    - previous_action (6D)
    
    Total: 26 dimensions
    """
    robot = env.scene["robot"]
    
    obs_list = [
        joint_pos_normalized(env),                          # 6D
        robot.data.joint_vel[:, :6],                        # 6D - noisy in real robot!
        robot.data.body_pos_w[:, robot.data.body_names.index("gripper_link"), :3],  # ee_pos_w (3D)
        goal_reach_error(env, command_name),                # 3D
        wrist_state(env),                                   # 1D
        jaw_state(env),                                     # 1D  
        previous_action(env),                               # 6D
    ]
    
    return torch.cat(obs_list, dim=-1)