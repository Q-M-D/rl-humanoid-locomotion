# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Mcm-Based-Rl-Direct-v0", help="Name of the task.")
parser.add_argument("--data_path", type=str, default='/home/mmlab-rl/codes/sensorimotor-rl/sensorimotor/data/forward_0.313.json', help="Path to the data file.")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import os
import yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.dict import print_dict
import mcm_based_rl.tasks
import numpy as np
import json




class PDController:
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        self.count = 0

    def get_actions(self, device):
        return_joing_pos = self.data['qpos'][self.count]
        self.count += 1
        if self.count >= len(self.data['qpos']):
            self.count = 0
        return torch.tensor(return_joing_pos, device=device)

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

controller = PDController(args_cli.data_path)


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(__file__), "..", "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        actions = controller.get_actions(env.unwrapped.device)

        # actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
        # apply actions
        obs_buf, _, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()