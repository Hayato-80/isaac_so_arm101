# SO-ARM101 Sim2Real Transfer Training Guide

Based on techniques from: https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41

## Overview (概要)

This guide explains the complete process of training SO-ARM101 in Isaac Lab and transferring learned policies to the real robot. ROS 2 is used as a common interface for seamless policy transfer between simulation and reality.

このドキュメントは、Isaac Lab で SO-ARM101 の強化学習を行い、学習結果を実機へ転送するまでの全プロセスを解説します。ROS 2 を共通インターフェースとして使用し、Sim と Real 間でポリシーの再利用を実現しています。

## Development Process (開発の流れ)

### 1. Isaac Lab Template-Based Project Setup

Create an independent SO101 project based on Isaac Lab's template, separate from the main IsaacLab repository. Install via pip:

```bash
python -m pip install -e /home/klab/ws/isaac_so_arm101
```

### 2. Implement SO101 Reach Task

Create a task for end-effector to reach a fixed target position. Initially, the observation space was **26-dimensional**:

**Initial Observation (26D)**:
- Normalized joint positions
- Joint velocity  
- EE absolute position
- Target delta
- Wrist/jaw state
- Previous action

### 3. ROS 2 Inference Node and Isaac Sim Integration

Create ROS 2 nodes to run policies on the real robot:

**Main files**:
```
scripts/ros2/policy_runner.py              # Main inference node
scripts/ros2/real_policy_runner.py         # Real robot policy runner  
scripts/ros2/joint_trajectory_to_array.py  # Command conversion adapter
```

**Policy Runner Responsibilities**:
1. Receive joint_states
2. Get EE position from TF transform
3. Assemble observations matching training format
4. Run inference with policy.onnx / policy.pt
5. Convert output actions to joint commands and publish

**Minimum required communication**:
- Publish JointState for 6 joints
- TF transform: base_link → gripper_frame_link
- Subscribe to policy's joint command targets

JointTrajectory adapter (when needed):
```bash
python3 scripts/ros2/joint_trajectory_to_array.py \
  --input-topic /isaac_joint_command \
  --output-topic /isaac_joint_command_array
```

### 4. Verify Communication in Simulation First

Before testing on real robot, verify wiring correctness in Isaac Sim:

**Verification points**:
- joint_states names and order are correct
- TF transform gets EE pose correctly
- Policy runner assembles observations properly
- Command topic receives data
- Robot joints actually move

### 5. Test on Real Robot

After Sim verification, connect the same policy runner to the real robot:

**Problem discovered**: The observation data used during training was completely different from what could be obtained from the real robot, causing the policy to output incorrect action values.

### 6. Reduce Observation Space from 26D to 17D

Revise observation design by removing unsuitable signals for sim2real:

**Removed items**:
- 6D joint velocity (noisy due to encoder noise on real robot)
- 3D EE absolute position ee_pos_w (not needed for fixed target tasks)

For fixed target tasks, what matters is "how far from the target" rather than "where am I now", so keep target_delta instead of absolute EE position.

**Final Observation (17D)**:
```python
obs = [
    joint_pos_normalized,   # 6D - normalized joint positions
    target_delta,           # 3D - difference from target
    wrist_roll,             # 1D - wrist roll position
    jaw,                    # 1D - gripper state
    prev_action,            # 6D - previous action for stability
]
```

This modification made policy deployment to the real robot much more stable.

### 7. Refine Reward Model and Response Characteristics

Address issues like Jaw opening tendency and Sim having better response than Real:

**Specific adjustments**:
- Change Jaw penalty from symmetric error to one-sided (open side only) - SO101 specific fix
- Expand action noise environment testing rate limit / tracking delay / transport lag
- Align policy cycle with physics cycle at 60 Hz

These are "why does Sim differ from Real" debugging tasks, not just "make it work".

## Key Sim2Real Techniques (主要な Sim2Real 技術)

### Observation Space Reduction (26D → 17D)

The most critical insight from the blog post is observation space reduction. The original 26-dimensional observation included signals that were noisy or irrelevant for real robot operation:

| Removed | Reason |
|---------|--------|
| joint_velocity (6D) | Encoder noise on real robot makes this unreliable |
| ee_pos_w (3D) | Absolute position not needed; target delta is sufficient |

### Domain Randomization (ドメインランダム化)

Train with physics variations to improve robustness:

```python
# Mass randomization ±30% (medium level)
mass_range = (0.7, 1.3)

# Friction range simulates different table surfaces  
friction_range = (0.5, 1.5)

# Joint damping/stiffness variations
stiffness_range = (0.5, 1.5)
damping_range = (0.5, 1.5)

# Actuation noise simulation
noise_std = 0.1  # 10% noise on commands
```

Three randomization levels are provided:
- **HIGH**: ±40% mass, ±70% stiffness/damping - most robust but slower convergence
- **MEDIUM** (default): ±30% mass, ±50% variations - recommended starting point
- **LOW**: ±10% mass, ±20% variations - for fine-tuning

### Simulation Frequency Matching

Set physics cycle to match real robot frequency (~60 Hz):

```python
self.sim.dt = 1.0 / 60.0  # 60 Hz simulation
```

## File Structure (ファイル構成)

```
src/isaac_so_arm101/tasks/reach/mdp/
├── sim2real_observations.py    # 17D observation functions
├── observations.py              # Original 26D observations (for reference)
└── rewards.py                   # Reward function definitions

src/isaac_so_arm101/tasks/reach/
├── sim2real_env_cfg.py          # Sim2Real environment config with randomization
├── sim2real_joint_pos_env_cfg.py # SO101-specific Sim2Real config
└── reach_env_cfg.py             # Base reach task configuration
```

## Usage (使用方法)

### Training Sim2Real Policy (Recommended Method)

Use the regular SO101 Reach task with the sim2real environment config override:

```bash
# Train with MEDIUM randomization (recommended starting point)
uv run train --task Isaac-SO-ARM101-Reach-v0 \
    +env.cfg_path=isaac_so_arm101.tasks.reach.sim2real_joint_pos_env_cfg.SoArm101ReachSim2RealEnvCfg_MEDIUM \
    --agent rsl_rl_ppo_cfg_entry_point --headless

# Train with HIGH randomization (more robust)
uv run train --task Isaac-SO-ARM101-Reach-v0 \
    +env.cfg_path=isaac_so_arm101.tasks.reach.sim2real_joint_pos_env_cfg.SoArm101ReachSim2RealEnvCfg_HIGH \
    --agent rsl_rl_ppo_cfg_entry_point --headless

# Train with LOW randomization (fine-tuning)
uv run train --task Isaac-SO-ARM101-Reach-v0 \
    +env.cfg_path=isaac_so_arm101.tasks.reach.sim2real_joint_pos_env_cfg.SoArm101ReachSim2RealEnvCfg_LOW \
    --agent rsl_rl_ppo_cfg_entry_point --headless

# Play trained policy without randomization
uv run play --task Isaac-SO-ARM101-Reach-v0 \
    +env.cfg_path=isaac_so_arm101.tasks.reach.sim2real_joint_pos_env_cfg.SoArm101ReachSim2RealEnvCfg_PLAY \
    --checkpoint path/to/policy.pt
```

### Real Robot Deployment

```bash
# Source ROS 2 workspace
source /home/klab/ws/isaac_so_arm101/install/setup.bash

# Run policy on real robot
ros2 run isaac_so101_ros2.policy_runner -p policy_path:=policy.onnx
```

## Results (結果)

### What Was Achieved

- Successfully learned SO101 reach task in Isaac Lab
- Established ROS 2 policy runner and Isaac Sim integration, returning joint commands from trained policies
- Confirmed real robot SO101 operation with stable behavior after observation space reduction to 17D

### Key Learnings

- For real robot operation, observation consistency between Sim and Real is crucial beyond just making things work in simulation
- Signals used for policy observations must have consistent meaning across both domains - careful verification required

## Simulation vs Reality Differences (Sim と Real の違い)

| Aspect | Simulation | Real Robot | Mitigation |
|--------|------------|------------|------------|
| Joint velocity noise | Low | High (encoder noise) | Remove from observations |
| Response speed | Fast | Slower | Domain randomization |
| Friction consistency | Perfect | Variable | Randomize friction |
| Mass properties | Fixed | Varying payloads | ±30% mass randomization |

## Training Strategy Recommendation

1. **Phase 1**: Start with HIGH randomization to learn robust policies (takes longer but more robust)
2. **Phase 2**: Fine-tune with MEDIUM randomization for better performance
3. **Phase 3**: Final fine-tuning with LOW randomization close to real robot conditions

This progressive curriculum approach helps the policy first learn to handle extreme variations, then refine its behavior as training progresses.

## Acknowledgments (謝辞)

This work is based on techniques from: https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41