# Sim2Real Reach Task - Quick Start Guide

Based on: https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41

## 🚀 Quick Start (3 Steps)

### Step 1: Train the Policy

```bash
# Train with medium domain randomization (recommended)
isaaclab-train.sh --task=SoArm101ReachSim2RealEnvCfg_MEDIUM --num_envs=4096
```

Wait for convergence (~10-20 minutes with 4096 parallel environments).

**Configuration options:**
- `_MEDIUM` (±30%): Recommended starting point
- `_HIGH` (±40%): More robust, slower
- `_LOW` (±10%): Fine-tuning only

### Step 2: Validate in Simulation

```bash
# Test the trained policy in clean simulation
isaaclab-train.sh \
  --task=SoArm101ReachSim2RealEnvCfg_PLAY \
  --checkpoint=/path/to/your/model.pt
```

Success rate should be **>95%** with reaching distance <1cm.

### Step 3: Deploy to Real Robot

```bash
# Terminal 1: Start policy runner
ros2 run isaac_so101_ros2 policy_runner \
  --ros-args \
  -p policy_path:=/path/to/policy.onnx

# Terminal 2 (optional): Start adapter if needed
ros2 run isaac_so101_ros2 joint_trajectory_to_array
```

## 📊 Key Observations (17D)

Your policy sees:

```
[joint_pos(6) | target_error(3) | wrist(1) | jaw(1) | prev_action(6)]
                └──────────┬──────────────┘
                           │
                    Focuses on what matters:
                    - WHERE am I? (joint_pos)
                    - HOW FAR from target? (error)
                    - Gripper state? (wrist, jaw)
                    - Smoothing (prev_action)
```

**NOT included (and why):**
- ❌ Joint velocities: Encoder noise ruins transfer
- ❌ Absolute EE position: Irrelevant for fixed targets
- ✅ Everything else: Essential for the task

## 🎯 Domain Randomization

Default: **±30% mass variation** (multiply each link mass by 0.7-1.3)

This simulates:
- Different payloads in gripper
- Manufacturing tolerances  
- Robot wear and tear

## ⚙️ Configuration Files

| File | Purpose |
|------|---------|
| `reach_env_cfg.py` | Base training environment |
| `sim2real_env_cfg.py` | Sim2real variant with 17D obs |
| `sim2real_joint_pos_env_cfg.py` | Joint-space control variant |
| `mdp/sim2real_observations.py` | 17D observation builder |
| `mdp/sim2real_randomization.py` | Domain randomization functions |

## 🔍 Troubleshooting

### "Policy is unstable"
→ Check observation order matches training. Verify joint names in URDF.

### "Real robot doesn't reach target"
→ Most likely: **encoder calibration** or **TF frame mismatch**
- Verify: `ros2 run tf2_tools view_frames.py`
- Check: Joint zero positions in URDF

### "Gripper opens when I don't want it to"
→ This is **intentional**! The policy was trained with asymmetric jaw penalty.
- Gripper naturally stays closed during reaching
- Expected behavior

## 📈 Expected Performance

| Stage | Success Rate | Reach Error |
|-------|--------------|-------------|
| Simulation (trained) | >99% | <1 cm |
| Simulation (domain rand) | >90% | <2 cm |
| Real robot* | 60-80% | 2-5 cm |

*Depends heavily on encoder accuracy and frame calibration

## 📚 Full Documentation

See [SIM2REAL_REACH_IMPLEMENTATION.md](./SIM2REAL_REACH_IMPLEMENTATION.md) for:
- Detailed architecture explanations
- ROS 2 integration guide
- Advanced customization
- References and citations

## 💡 Key Insights from Original Paper

1. **Less is more**: Removing noisy signals improves transfer
2. **Error > Position**: Use goal error, not absolute positions
3. **Domain randomization works**: ±30% mass is surprisingly effective
4. **Gripper control is learnable**: Asymmetric penalty prevents opening

## 🛠️ Hardware Requirements (Real Robot)

- Joint state publishers at 60 Hz
- TF broadcasting (base_link → gripper_frame_link)
- Policy inference capable of 60 Hz (even on CPU)

## 📝 Environment Variables

Set these in your shell:

```bash
# Path to trained model (ONNX or PyTorch)
export POLICY_MODEL=/path/to/policy.onnx

# Target position (robot base frame)
export TARGET_X=-0.15
export TARGET_Y=-0.20
export TARGET_Z=0.20
```

Then launch:

```bash
ros2 run isaac_so101_ros2 policy_runner \
  --ros-args \
  -p policy_path:=$POLICY_MODEL \
  -p target_x:=$TARGET_X \
  -p target_y:=$TARGET_Y \
  -p target_z:=$TARGET_Z
```

## 🎓 Learning Path

1. **Read** the original blog (Japanese): https://qiita.com/takahasr_rs/items/ce085a2b898218f9be41
2. **Train** with MEDIUM configuration
3. **Test** in PLAY environment
4. **Deploy** to real robot
5. **Debug** any differences
6. **Read** the full implementation guide

## ✨ Success Criteria

Your implementation is working when:

✅ Simulation (no randomization) reaches target >99% of the time
✅ Simulation (±30% randomization) reaches target >90% of the time  
✅ Real robot reaches target 60-80% of the time (with good calibration)
✅ Gripper stays closed unless needed
✅ All motion is smooth (no jerking)

Good luck with your sim2real deployment! 🤖
