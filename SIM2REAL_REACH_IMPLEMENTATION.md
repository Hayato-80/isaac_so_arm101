# SO101 Sim2Real Reach Task Implementation

This implementation is based on the techniques from the blog post: https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41

A comprehensive guide to implementing sim2real reach tasks for the SO-ARM101 robot using Isaac Lab and ROS 2.

## Overview

This module provides a complete implementation of a sim2real reach task (moving the end-effector to a target position) for the SO-ARM101 robot. The key innovation is a **17-dimensional observation space** optimized for transfer from simulation to the real robot.

## Key Contributions from the Reference Article

### 1. **Observation Dimension Reduction (26D → 17D)**

The original 26D observation included:
- Joint positions (6D)
- Joint velocities (6D) ❌ **Noisy in real robot**
- End-effector absolute position (3D) ❌ **Not needed for fixed targets**
- Target delta (3D)
- Wrist/jaw state (2D)
- Previous action (6D)

The optimized **17D observation** removes noisy signals:

```python
obs = [
    joint_pos_normalized(6),      # Normalized joint positions [-1, 1]
    target_delta(3),              # (target_pos - current_ee_pos)
    wrist_roll(1),                # Wrist roll joint state
    jaw(1),                        # Gripper position
    prev_action(6),               # Last action for stability
]
```

**Why this matters:** Joint velocities from encoders are very noisy in real robots. The policy doesn't need absolute EE position for fixed-target tasks—only the error matters.

### 2. **Domain Randomization for Robustness**

- **Robot mass ±30%** (default): Simulates payload variations
- **Joint friction/damping variations**: Different lubrication states
- **Action noise**: Motor command imprecision
- **Tracking delay simulation**: Control latency

### 2b. **Pink Noise Action Randomization**

Pink noise is a correlated action perturbation used during training to make the policy more robust to realistic actuator behavior.

How it works in this code:

1. A per-environment pink-noise state is stored in the environment object.
2. On reset, the state is cleared for the reset environments.
3. On each `before_step`, the code samples white noise and passes it through several leaky integrators with different smoothing factors.
4. Those filtered signals are mixed together and added to the action.
5. The final action is clamped to the valid range.

Why it is useful:

- Gaussian noise is memoryless and can look jittery.
- Pink noise is temporally correlated and smoother.
- That smoother structure often better matches real actuator drift, small latency, and command imperfections.

Important training note:

- Pink noise is enabled in the training sim2real configs.
- It is disabled in the PLAY/inference configs so the policy output stays deterministic when you deploy or evaluate it.

### 3. **Asymmetric Jaw Penalty**

The gripper tends to open too easily. Solution:
- **Heavy penalty for opening** (jaw > 0.01)
- **No penalty for closing** (jaw ≤ 0.01)

This encourages the policy to keep the gripper closed during manipulation.

### 4. **60 Hz Control Frequency**

Ensures simulation matches real robot timing:
```python
self.sim.dt = 1.0 / 60.0
self.decimation = 1  # No decimation at 60 Hz
```

## File Structure

```
src/isaac_so_arm101/
├── tasks/reach/
│   ├── reach_env_cfg.py              # Base reach environment configuration
│   ├── sim2real_env_cfg.py            # Sim2real variant (extends ReachEnvCfg)
│   ├── sim2real_joint_pos_env_cfg.py  # Joint position control variant
│   └── mdp/
│       ├── observations.py            # Standard observation functions
│       ├── sim2real_observations.py   # Optimized 17D observations
│       ├── sim2real_randomization.py  # Domain randomization functions
│       ├── rewards.py                 # Reward functions (includes jaw penalty)
│       └── terminations.py
└── scripts/ros2/sim2real_so101/
    ├── policy_runner.py              # ROS 2 node for policy inference
    └── joint_trajectory_to_array.py  # Message adapter
```

## Usage

### Training a Sim2Real Policy

#### 1. **Start with Medium Randomization (Recommended)**

```bash
# Train with medium domain randomization (±30% mass variation)
isaaclab-train.sh \
  --task=SoArm101ReachSim2RealEnvCfg_MEDIUM \
  --checkpoint=/path/to/checkpoint \
  --num_envs=4096
```

**Configuration variants:**
- `SoArm101ReachSim2RealEnvCfg_MEDIUM`: ±30% mass (recommended starting point)
- `SoArm101ReachSim2RealEnvCfg_HIGH`: ±40% mass (more robust but slower training)
- `SoArm101ReachSim2RealEnvCfg_LOW`: ±10% mass (fine-tuning after rough training)
- `SoArm101ReachSim2RealEnvCfg_PLAY`: No randomization (inference only)

#### 2. **Verify in Simulation**

```bash
# Play/inference in simulation (no domain randomization)
isaaclab-train.sh \
  --task=SoArm101ReachSim2RealEnvCfg_PLAY \
  --checkpoint=/path/to/trained/model
```

#### 3. **Deploy to Real Robot**

```bash
# Terminal 1: Start Isaac Sim with ROS 2 bridge
# (Assumes Isaac Sim ROS 2 extension is enabled)
isaac_sim_path/isaac-sim.sh --physics-engine=physx

# Terminal 2: Start policy runner
ros2 run isaac_so101_ros2 policy_runner \
  --ros-args \
  -p policy_path:=/path/to/policy.onnx \
  -p target_x:=-0.15 \
  -p target_y:=-0.20 \
  -p target_z:=0.20

# Terminal 3 (if using JointTrajectory adapter):
ros2 run isaac_so101_ros2 joint_trajectory_to_array \
  --ros-args \
  -p input_topic:=/isaac_joint_command \
  -p output_topic:=/isaac_joint_command_array

# Alternative (Launch File with direct CSV and TF frames logging):
ros2 launch so101_bringup policy_runner.launch.py \
  hardware_type:=real \
  run_bridge:=true \
  arm_controller:=trajectory_controller \
  bridge_command_type:=trajectory \
  policy_state_topic:=/follower/joint_states \
  policy_command_topic:=/isaac_joint_commands \
  policy_path:=/home/klab/ws/isaac_ros-dev/src/isaac_so101_ros2/policy/1_f_action_and_DR/action_domain/policy.pt \
  env_path:=/home/klab/ws/isaac_ros-dev/src/isaac_so101_ros2/policy/1_f_action_and_DR/action_domain/env.yaml \
  policy_target_xyz:="[0.2, 0.2, 0.3]" \
  run_command_adapter:=false \
  policy_log_csv_path:=/home/klab/ws/isaac_ros-dev/timeseries_test.csv \
  policy_base_frame:=follower/base_link \
  policy_ee_frame:=follower/gripper_frame_link
```

## ROS 2 Integration

### Required Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/isaac_joint_states` | `sensor_msgs/JointState` | ← (Input) | Current joint positions from robot |
| `/isaac_joint_command` | `sensor_msgs/JointState` | → (Output) | Target joint positions from policy |
| `/isaac_joint_command_array` | `std_msgs/Float64MultiArray` | → (Output) | Alternative command format (if using adapter) |

### Required Transforms

| Frame Pair | Description |
|-----------|-------------|
| `base_link` → `gripper_frame_link` | End-effector position (for goal error calculation) |

**CRITICAL:** The frame names and joint order must exactly match your URDF!

### Policy Runner Configuration

```yaml
policy_runner:
  ros__parameters:
    policy_path: "policy.onnx"           # ONNX or PyTorch model
    obs_dim: 17                           # Observation dimension
    action_dim: 6                         # Action dimension
    control_freq: 60.0                    # Hz
    target_x: -0.15                       # Target position (robot base frame)
    target_y: -0.20
    target_z: 0.20
```

## Observations in Detail

### 1. Normalized Joint Positions (6D)

Based on SO101 typical ranges:

```python
# Shoulder Pan: ±2.88 rad (~±165°)
# Shoulder Lift: ±2.69 rad (~±154°)
# Elbow: 0 to 3.05 rad (0-175°)
# Wrist Pitch: ±2.69 rad (~±154°)
# Wrist Roll: ±3.05 rad (~±175°)
# Jaw: 0 to 0.04 m (gripper opening)
```

Normalized to [-1, 1] range:
```
normalized = 2 * (value - min) / (max - min) - 1
```

### 2. Goal Reach Error (3D)

The key insight: **use error, not absolute position**

```python
error = target_position - current_ee_position
```

This is invariant to different picking locations and is what the policy actually needs to minimize.

### 3. Wrist Roll & Jaw State (2D)

Explicit joint states for gripper control:
```python
wrist_roll = robot.data.joint_pos[:, wrist_roll_index]
jaw = robot.data.joint_pos[:, jaw_index]
```

### 4. Previous Action (6D)

For policy stability and to help handle delays:
```python
prev_action = last_action_from_previous_step
```

## Domain Randomization Details

### Robot Mass Randomization

The **most important** randomization for sim2real transfer:

```python
# Default: ±30% (0.7 to 1.3 multiplier)
mass_randomized = original_mass * uniform(0.7, 1.3)
```

This simulates:
- Different payloads in the gripper
- Manufacturing tolerances
- Tool changes

### Expected Impact

| Randomization | Effect |
|---------------|--------|
| **No randomization** | Perfect in sim, fails on real robot |
| **±10% mass** | Minimal robustness, good for fine-tuning |
| **±30% mass** | Good balance (recommended) |
| **±40% mass** | Very robust but slower training |

## Troubleshooting

### Issue: Policy Actions Are Unstable

**Solution:** Check observation order matches training:
1. Verify joint names in URDF match `joint_names` in policy_runner.py
2. Verify TF frames exist: `ros2 run tf2_tools view_frames.py`
3. Verify observation dimensions: 6+3+1+1+6 = 17

### Issue: Gripper Opens When It Shouldn't

**Solution:** The jaw asymmetry penalty is working correctly. This is intentional behavior learned during training. If undesired:
1. Increase jaw penalty weight in rewards config
2. Retrain with higher jaw penalty
3. Or adjust target position to avoid gripper contact

### Issue: Real Robot Doesn't Follow Simulation

**Most Common Causes:**
1. **Joint velocity noise:** This observation was removed for good reason! Don't add it back.
2. **Frame misalignment:** Verify TF transforms match URDF exactly
3. **Frequency mismatch:** Ensure both simulation and real robot run at 60 Hz
4. **Encoder calibration:** Real robot encoders must match URDF zero positions

## Advanced: Adapting to Your Robot

### Step 1: Customize Joint Ranges

Edit [sim2real_observations.py](./mdp/sim2real_observations.py):

```python
joint_ranges = torch.tensor([
    [your_max_1, your_min_1],   # Joint 1
    [your_max_2, your_min_2],   # Joint 2
    # ... etc
], device=joint_pos.device)
```

### Step 2: Verify Command Target Ranges

Edit [reach_env_cfg.py](./reach_env_cfg.py):

```python
ee_pose = mdp.UniformPoseCommandCfg(
    ranges=mdp.UniformPoseCommandCfg.Ranges(
        pos_x=(-0.1, 0.1),   # Your reachable workspace
        pos_y=(-0.25, -0.1),
        pos_z=(0.1, 0.3),
    ),
)
```

### Step 3: Train and Validate

```bash
# Step 1: Train with medium randomization
isaaclab-train.sh --task=SoArm101ReachSim2RealEnvCfg_MEDIUM --num_envs=4096

# Step 2: Fine-tune with low randomization
isaaclab-train.sh \
  --task=SoArm101ReachSim2RealEnvCfg_LOW \
  --checkpoint=/path/to/medium/model

# Step 3: Verify in simulation
isaaclab-train.sh \
  --task=SoArm101ReachSim2RealEnvCfg_PLAY \
  --checkpoint=/path/to/finetuned/model

# Step 4: Deploy to real robot
# Use policy_runner.py via ROS 2
```

## Performance Expectations

### Simulation Performance (No Randomization)

- Success Rate: 95-99%
- Convergence: 5-10k environment steps (with 4096 parallel envs)
- Average Reaching Distance: < 0.01 m

### Real Robot Performance (After Sim2Real)

- Success Rate: 60-80% (depends on real robot accuracy)
- Reaching Distance: 0.02-0.05 m (dependent on encoder calibration)
- Gripper Control: Keep closed (intentional by design)

## References

1. **Original Blog Post:** https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41
2. **Isaac Lab Documentation:** https://docs.isaacsim.org/
3. **ROS 2 Documentation:** https://docs.ros.org/

## Key Takeaways

1. **17D observations are better than 26D** for real robot transfer
2. **Remove noisy signals** (joint velocities) even if they seem useful
3. **Use error-based observations** instead of absolute positions for fixed-target tasks
4. **Domain randomization is critical** - ±30% mass variation is a good starting point
5. **Asymmetric jaw penalty** prevents unwanted gripper opening
6. **60 Hz control frequency** must match across sim and real robot
7. **Frame alignment is crucial** - verify TF transforms and URDF exactly match

## Contributing

Found an issue or have improvements? The sim2real transfer process is still an active research area with many unknowns!

## License

BSD-3-Clause (see LICENSE file)
