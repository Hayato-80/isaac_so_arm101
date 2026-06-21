# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint with a user-specified target pose."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Verify a trained RL agent with a custom target pose.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--log_csv", action="store_true", default=False, help="Log joint position, velocity, EE position, and target position to CSV.")
parser.add_argument("--csv_filename", type=str, default="play_custom_log.csv", help="Filename for the logged CSV.")

# Custom target arguments
parser.add_argument(
    "--target_pos",
    type=float,
    nargs=3,
    default=[0.15, 0.0, 0.2],
    help="Target position [x, y, z] in robot base frame (meters). Default: 0.15 0.0 0.2",
)
parser.add_argument(
    "--target_rpy",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.0],
    help="Target orientation [roll, pitch, yaw] in radians. Default: 0.0 0.0 0.0",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent and custom target pose."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Override target position/orientation ranges to be exact single values
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "ee_pose"):
        tx, ty, tz = args_cli.target_pos
        tr, tp, tyaw = args_cli.target_rpy
        
        env_cfg.commands.ee_pose.ranges.pos_x = (tx, tx)
        env_cfg.commands.ee_pose.ranges.pos_y = (ty, ty)
        env_cfg.commands.ee_pose.ranges.pos_z = (tz, tz)
        env_cfg.commands.ee_pose.ranges.roll = (tr, tr)
        env_cfg.commands.ee_pose.ranges.pitch = (tp, tp)
        env_cfg.commands.ee_pose.ranges.yaw = (tyaw, tyaw)
        
        print(f"[INFO] Configured environment target position to: {args_cli.target_pos}")
        print(f"[INFO] Configured environment target orientation (RPY) to: {args_cli.target_rpy}")

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Set up CSV logging if enabled
    csv_file = None
    csv_writer = None
    if args_cli.log_csv:
        csv_path = os.path.join(log_dir, args_cli.csv_filename)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        csv_file = open(csv_path, mode="w", newline="")
        csv_writer = csv.writer(csv_file)
        
        # Build headers dynamically from the robot articulation
        robot = env.unwrapped.scene["robot"]
        joint_names = robot.data.joint_names
        
        headers = ["time", "step"]
        # position headers
        for name in joint_names:
            headers.append(f"{name}_pos")
        # velocity headers
        for name in joint_names:
            headers.append(f"{name}_vel")
        # EE position headers
        headers.extend(["ee_pos_x", "ee_pos_y", "ee_pos_z"])
        # Target position headers (if ee_pose command is available)
        if "ee_pose" in env.unwrapped.command_manager.active_terms:
            headers.extend(["target_pos_x", "target_pos_y", "target_pos_z"])
        csv_writer.writerow(headers)
        print(f"[INFO] Logging data to CSV: {csv_path}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    start_wall_time = time.time()
    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, _, _ = env.step(actions)
                
                # Log data if CSV logging is enabled
                if csv_writer is not None:
                    robot = env.unwrapped.scene["robot"]
                    # get data for the first environment (env 0)
                    j_pos = robot.data.joint_pos[0].cpu().numpy()
                    j_vel = robot.data.joint_vel[0].cpu().numpy()
                    
                    # Find gripper_link index
                    if "gripper_link" in robot.data.body_names:
                        ee_idx = robot.data.body_names.index("gripper_link")
                        ee_pos = robot.data.body_pos_w[0, ee_idx].cpu().numpy()
                    else:
                        ee_pos = [0.0, 0.0, 0.0]
                    
                    elapsed_time = time.time() - start_wall_time
                    row = [elapsed_time, timestep]
                    row.extend(j_pos.tolist())
                    row.extend(j_vel.tolist())
                    row.extend(ee_pos.tolist())
                    
                    # Target position (if ee_pose command is available)
                    if "ee_pose" in env.unwrapped.command_manager.active_terms:
                        command = env.unwrapped.command_manager.get_command("ee_pose")
                        from isaaclab.utils.math import combine_frame_transforms
                        # Transform target from base frame to world frame
                        target_pos_w, _ = combine_frame_transforms(
                            robot.data.root_pos_w[:, :3], robot.data.root_quat_w, command[:, :3]
                        )
                        target_pos = target_pos_w[0].cpu().numpy().tolist()
                        row.extend(target_pos)
                    
                    csv_writer.writerow(row)
                    csv_file.flush()  # Force writing to disk immediately
                    
            timestep += 1
            if args_cli.video and timestep == args_cli.video_length:
                break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("[INFO] Play loop interrupted by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
    finally:
        # close the CSV file if open
        if csv_file is not None:
            csv_file.close()
            print(f"[INFO] Saved CSV log to {csv_path}")
        # close the simulator
        env.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    # run the main function
    main()
