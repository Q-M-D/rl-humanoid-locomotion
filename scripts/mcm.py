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
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

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
import mcm_based_rl.tasks
from mbc import MBC, walk_controller
import numpy as np


with open(os.path.join(os.path.dirname(__file__), "config_mujoco.yaml"), "r") as f:
    config = yaml.safe_load(f)
args_cli.config = config

args_cli.scale = 0

def init_mbc():
    mjcf_folder = args_cli.config['mjcf_folder']
    modelfilename_pin = os.path.join(mjcf_folder, "dora2_stand_fix_pin.xml")

    mbcs = []
    for _ in range(args_cli.num_envs):
        mbc = MBC(modelfilename_pin)
        mbcs.append(mbc)
    return mbcs

def mbc_get_actions(env, mbc):
    """
    Get actions from mbc.
    ---
    input: `env` - environment
    input: `mbc` - mbc object
    ---
    output: actions
    ---
    mbc need: MBC, ppos, pvel, contact_mask
    """
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    # return torch.ones(env.action_space.shape, device=env.unwrapped.device) * args_cli.scale
    ppos = env.env.get_pin_q()
    pvel = env.env.get_pin_qvel()
    contact_mask = env.env.get_contact_mask()
    for i in range(args_cli.num_envs):
        # get mbc
        mbc_i = mbc[i]
        # get state
        # get actions
        controller_output = walk_controller(
            mbc_i, 
            np.array(ppos[i].cpu()), 
            np.array(pvel[i].cpu()), 
            np.array(contact_mask[i].cpu())
        )
        controller_tensor = torch.as_tensor(controller_output, dtype=torch.float32, device=env.unwrapped.device)
        actions[env.action_space.shape[1]*i:env.action_space.shape[1]*(i+1)] = controller_tensor
    return actions

def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # initialize mbc
    mbc = init_mbc()

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        actions = mbc_get_actions(env, mbc)
        
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