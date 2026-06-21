# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SO101 Sim2Real Domain Randomization Configuration

Based on techniques from: https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41

This module provides domain randomization configuration for sim2real transfer.
It leverages Isaac Lab's built-in randomization functions for:
- Mass randomization: isaaclab.envs.mdp.randomize_rigid_body_mass
- Joint parameters: isaaclab.envs.mdp.randomize_joint_parameters
- Gravity variation: isaaclab.envs.mdp.randomize_physics_scene_gravity

Key sim2real domain randomization strategies:
1. Mass randomization - payload and manufacturing tolerance variations (CRITICAL)
2. Joint parameter randomization - friction and damping variations
3. Gravity randomization - slight variations in gravitational acceleration
4. Action noise - simulates command noise in real robot
5. Pink noise - correlated command noise with low-frequency structure

These randomizations help policies generalize better from simulation to reality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, noise as noise_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class Sim2RealRandomizationCfg:
    """
    Configuration for sim2real domain randomization.
    
    Uses Isaac Lab's built-in randomization functions where possible.
    Attributes control which randomizations are enabled and their parameters.
    """
    
    # Mass randomization (MOST CRITICAL for sim2real transfer)
    # Simulates payload variations and manufacturing tolerances
    randomize_mass: bool = True
    mass_range: tuple[float, float] = (0.7, 1.3)  # ±30% variation
    
    # Joint parameter randomization (friction and damping)
    # Simulates variations in joint lubrication and motor response
    randomize_joint_params: bool = True
    friction_range: tuple[float, float] = (0.5, 1.5)  # friction multiplier
    damping_range: tuple[float, float] = (0.5, 1.5)  # damping multiplier
    
    # Gravity randomization (minor but adds robustness)
    # Simulates slight variations in perceived gravity
    randomize_gravity: bool = False
    gravity_range: tuple[list[float], list[float]] = (
        [-0.1, -0.1, -0.1],  # min offset
        [0.1, 0.1, 0.1],      # max offset
    )


def randomize_action_noise(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """
    Add Gaussian noise to joint actions (custom sim2real function).
    
    Simulates motor command noise and control imperfections in real robots.
    This function modifies the action tensor in place before physics application.
    
    Args:
        env: The environment instance.
        env_ids: Indices of environments to randomize.
        std: Standard deviation of noise. Defaults to 0.05.
        asset_cfg: Scene entity configuration for robot.
        
    Example:
        >>> # Use in EventTermCfg with mode="before_step"
        >>> EventTermCfg(
        ...     func=randomize_action_noise,
        ...     mode="before_step",
        ...     params={"std": 0.05},
        ... )
    """
    # Add Gaussian noise to actions
    noise = torch.randn_like(env.action_manager.action[env_ids]) * std
    env.action_manager.action[env_ids] = torch.clamp(
        env.action_manager.action[env_ids] + noise, -1.0, 1.0
    )


def generate_pink_noise(size: list[int] | tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate unit-variance pink noise using the Timmer & Koenig FFT algorithm in PyTorch."""
    samples = size[-1]
    
    # Calculate frequencies (Hermitian spectrum for real output)
    f = torch.fft.rfftfreq(samples, device=device)
    
    # Low-frequency cutoff
    fmin = 1.0 / samples
    
    # Build scaling factors for frequencies: S(f) = 1 / f
    # For amplitude scale, we take the square root of PSD: f^(-exponent/2) with exponent=1
    s_scale = f.clone()
    cutoff_idx = torch.sum(s_scale < fmin).item()
    if cutoff_idx > 0 and cutoff_idx < len(s_scale):
        s_scale[:cutoff_idx] = s_scale[cutoff_idx]
    s_scale = s_scale ** -0.5
    
    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].clone()
    w[-1] *= (1 + (samples % 2)) / 2.0
    sigma = 2.0 * torch.sqrt(torch.sum(w ** 2)) / samples
    
    # Adjust size to generate one complex Fourier component per frequency
    noise_size = list(size)
    noise_size[-1] = len(f)
    
    # Generate scaled random power + phase
    s_scale_broadcast = s_scale.view(*([1] * (len(size) - 1)), -1)
    sr = torch.randn(noise_size, device=device, dtype=dtype) * s_scale_broadcast
    si = torch.randn(noise_size, device=device, dtype=dtype) * s_scale_broadcast
    
    # If the signal length is even, Nyquist frequency coefficient must be real
    if not (samples % 2):
        si[..., -1] = 0.0
        sr[..., -1] *= 1.4142135623730951  # sqrt(2) to fix magnitude
        
    # DC component must be real
    si[..., 0] = 0.0
    sr[..., 0] *= 1.4142135623730951  # sqrt(2) to fix magnitude
    
    # Combine power + phase to complex Fourier components
    s = torch.complex(sr, si)
    
    # Transform to real time series & scale to unit variance
    y = torch.fft.irfft(s, n=samples, dim=-1) / sigma
    
    return y


class TorchPinkNoiseProcess:
    """Vectorized Pink Noise Process implemented in PyTorch."""

    def __init__(self, size: tuple[int, ...], device: torch.device, dtype: torch.dtype, scale: float = 1.0):
        self.size = list(size)
        self.device = device
        self.dtype = dtype
        self.scale = scale
        self.time_steps = self.size[-1]
        self.buffer = None
        self.idx = 0
        self.reset()

    def reset(self):
        self.buffer = generate_pink_noise(self.size, self.device, self.dtype)
        self.idx = 0

    def reset_env_ids(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        # Generate new noise for the reset environments only
        sub_size = list(self.size)
        sub_size[0] = env_ids.numel()
        
        new_noise = generate_pink_noise(sub_size, self.device, self.dtype)
        self.buffer[env_ids] = new_noise

    def sample(self, T: int = 1) -> torch.Tensor:
        n = 0
        ret = []
        while n < T:
            if self.idx >= self.time_steps:
                self.reset()
            m = min(T - n, self.time_steps - self.idx)
            ret.append(self.buffer[..., self.idx : (self.idx + m)])
            n += m
            self.idx += m

        ret = torch.cat(ret, dim=-1)
        if T == 1:
            ret = ret.squeeze(-1)
        return self.scale * ret


def reset_pink_noise_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    num_scales: int = 4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the stored pink-noise filter state for selected environments."""
    del num_scales, asset_cfg
    process = getattr(env, "_action_pink_noise_process", None)
    if process is not None:
        process.reset_env_ids(env_ids)


def randomize_action_pink_noise(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    std: float = 0.05,
    num_scales: int = 4,
    alpha_min: float = 0.75,
    alpha_max: float = 0.98,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Add pink action noise to joint actions using TorchPinkNoiseProcess."""
    del num_scales, alpha_min, alpha_max, asset_cfg
    action = getattr(env.action_manager, "action", None)
    if action is None or env_ids.numel() == 0:
        return

    seq_len = getattr(env, "max_episode_length", 1000)
    if seq_len is None or seq_len <= 0:
        seq_len = 1000

    process = getattr(env, "_action_pink_noise_process", None)
    if process is None or process.size[0] != env.num_envs or process.size[1] != action.shape[-1]:
        process = TorchPinkNoiseProcess(
            size=(env.num_envs, action.shape[-1], seq_len),
            device=action.device,
            dtype=action.dtype,
            scale=1.0
        )
        setattr(env, "_action_pink_noise_process", process)

    # Sample 1 step of noise. Returns shape (num_envs, action_dim)
    noise_torch = process.sample(T=1)

    # Apply noise only to selected env_ids
    env.action_manager.action[env_ids] = torch.clamp(
        env.action_manager.action[env_ids] + std * noise_torch[env_ids], -1.0, 1.0
    )


class PinkNoiseObservationModel(noise_utils.NoiseModel):
    """Pink noise model for observation corruption with temporal correlation."""

    def __init__(self, noise_model_cfg: "PinkNoiseObservationModelCfg", num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self._std = noise_model_cfg.std
        self._process: TorchPinkNoiseProcess | None = None

    def reset(self, env_ids=None):
        if self._process is None:
            return
        if env_ids is None:
            self._process.reset()
        else:
            self._process.reset_env_ids(env_ids)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        obs_dim = data.shape[1] if data.dim() > 1 else 1
        if self._process is None or self._process.size[0] != data.shape[0] or self._process.size[1] != obs_dim:
            self._process = TorchPinkNoiseProcess(
                size=(data.shape[0], obs_dim, 1000),
                device=data.device,
                dtype=data.dtype,
                scale=self._std
            )

        # Sample 1 step of noise
        noise_torch = self._process.sample(T=1)

        if data.dim() == 1:
            noise_torch = noise_torch.squeeze(-1)

        return data + noise_torch


@configclass
class PinkNoiseObservationModelCfg(noise_utils.NoiseModelCfg):
    """Configuration for pink noise observation model."""

    class_type: type = PinkNoiseObservationModel
    noise_cfg: noise_utils.NoiseCfg = noise_utils.ConstantNoiseCfg(bias=0.0)

    std: float = 0.02
    num_scales: int = 4
    alpha_min: float = 0.75
    alpha_max: float = 0.98
