# Building SO101 ROS 2 Package from isaac_so_arm101 Project

This guide explains how to build the ROS 2 package within your `isaac_so_arm101` project structure.

## Prerequisites

```bash
# Install required dependencies (Ubuntu 24.04)
sudo apt update
sudo apt install -y python3-colcon-common-extensions \
    ros-jazzy-rclpy \
    ros-jazzy-sensor-msgs \
    ros-jazzy-std-msgs \
    ros-jazzy-geometry-msgs \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-trajectory-msgs

# Install ONNX runtime for Python (for policy inference)
pip install onnxruntime
```

## Build the ROS 2 Package (Direct from isaac_so_arm101)

The ROS 2 package is located at `src/isaac_so_arm101/scripts/ros2/sim2real_so101`.

### Step 1: Remove COLCON_IGNORE (if present)

```bash
cd /home/klab/ws/isaac_so_arm101
rm -f src/isaac_so_arm101/scripts/ros2/sim2real_so101/COLCON_IGNORE
```

### Step 2: Source ROS 2 Environment

```bash
source /opt/ros/jazzy/setup.bash
```

### Step 3: Build with Colcon

```bash
cd /home/klab/ws/isaac_so_arm101
colcon build --packages-select isaac_so101_ros2
```

**Output:**
```
Starting >>> isaac_so101_ros2
Finished <<< isaac_so101_ros2 [0.67s]
Summary: 1 package finished [0.76s]
```

### Step 4: Source the Built Workspace

```bash
source /home/klab/ws/isaac_so_arm101/install/setup.bash
```

## Running the ROS 2 Nodes

### Run Policy Runner Node

```bash
ros2 run isaac_so101_ros2.policy_runner \
    --ros-args -p policy_path:=policy.onnx \
                       -p obs_dim:=17 \
                       -p action_dim:=6 \
                       -p control_freq:=60.0
```

### Run Trajectory Adapter Node (Optional)

```bash
ros2 run isaac_so101_ros2.joint_trajectory_to_array \
    --ros-args -p input_topic:=/isaac_joint_command \
                       -p output_topic:=/isaac_joint_command_array
```

## Package Structure

The ROS 2 package is located at:
```
src/isaac_so_arm101/scripts/ros2/sim2real_so101/
├── CMakeLists.txt          # CMake build configuration
├── package.xml             # Package metadata and dependencies
├── setup.py                # Python setup script
├── __init__.py             # Python package marker
├── resource/               # ROS 2 resource index
│   └── isaac_so101_ros2    # Resource file for ament
├── policy_runner.py        # Main policy inference node (60Hz, ONNX support)
└── joint_trajectory_to_array.py  # Command adapter for simulation
```

## Verification Commands

After building, verify the installation:

```bash
# List available nodes
ros2 node list

# Check package location
ros2 pkg prefix isaac_so101_ros2

# Test running with help
ros2 run isaac_so101_ros2.policy_runner --help
```

## Common Issues and Solutions

### Issue: COLCON_IGNORE prevents building

**Solution:** Remove the COLCON_IGNORE file from the package directory.

```bash
rm src/isaac_so_arm101/scripts/ros2/sim2real_so101/COLCON_IGNORE
```

### Issue: ROS 2 environment not sourced

**Solution:** Source your ROS 2 setup script before running commands.

```bash
source /opt/ros/jazzy/setup.bash
```

### Issue: Module import errors in policy_runner.py

**Solution:** Ensure ONNX runtime is installed.

```bash
pip install onnxruntime
```

## Training Commands (Quick Reference)

```bash
# Reach task - Sim2Real training
uv run train --task SO-ARM101-Reach-Sim2Real-v0 \
    --agent reach_sim2real_medium_cfg_entry_point \
    --num_envs 4096 \
    --max_iterations 1500

# Lift task - Sim2Real training  
uv run train --task SO-ARM101-Lift-Sim2Real-v0 \
    --agent lift_sim2real_medium_cfg_entry_point \
    --num_envs 4096 \
    --max_iterations 1500

# Play trained policy in simulation
uv run play --task SO-ARM101-Lift-Sim2Real-Play-v0 \
    --load_run logs/rsl_rl/lift \
    --checkpoint 99.pth
```

## References

- [ROS 2 Documentation](https://docs.ros.org/en/jazzy/)
- [Colcon Build Tool](https://colcon.readthedocs.io/)
- [Isaac Lab Sim2Real Guide](README_SIM2REAL.md)