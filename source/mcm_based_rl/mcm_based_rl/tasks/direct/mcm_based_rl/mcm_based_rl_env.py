# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ImuCfg

from .mcm_based_rl_env_cfg import McmBasedRlEnvCfg
from .trans import transform_points
import os
import yaml


class McmBasedRlEnv(DirectRLEnv):
    cfg: McmBasedRlEnvCfg

    def __init__(self, cfg: McmBasedRlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        """12 joints in total.
        l_hip_yaw, l_hip_pitch, l_hip_roll, l_knee, l_ankle_pitch, l_ankle_roll
        r_hip_yaw, r_hip_pitch, r_hip_roll, r_knee, r_ankle_pitch, r_ankle_roll
        """
        self._joint_dof_idx = [
            self.robot.find_joints(dof_name)[0] for dof_name in self.cfg.controllable_dof_names
        ]
        self.l_roll_joint_idx = self.robot.find_joints("l_leg_ankle_roll_joint")[0]
        self.r_roll_joint_idx = self.robot.find_joints("r_leg_ankle_roll_joint")[0]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        self.custom_cfg = self.custom_config()
        self.sensor_relative_pos = torch.tensor(self.custom_cfg['sensor_pos'], dtype=torch.float32, device=self.device)
        self.velocity_input = torch.tensor(0.313, dtype=torch.float32, device=self.device)
        self.single_step_obs = self.custom_cfg['single_step_observation']
        self.obs_his_len = self.custom_cfg['observation_history_len']
        self.actions_dim = self.custom_cfg['actions_dim']
        self.vx_lambda = torch.tensor(self.custom_cfg['vx_lambda'], dtype=torch.float32, device=self.device)
        self.vy_lambda = torch.tensor(self.custom_cfg['vy_lambda'], dtype=torch.float32, device=self.device)
        self.vz_lambda = torch.tensor(self.custom_cfg['vz_lambda'], dtype=torch.float32, device=self.device)
        self.wz_lambda = torch.tensor(self.custom_cfg['wz_lambda'], dtype=torch.float32, device=self.device)
        self.super_obs_buf = self.init_buffer()
        self.buffer_index = 0
        self.last_actions = torch.zeros(self.num_envs, self.actions_dim, device=self.device)

        self.obs = None

    def custom_config(self):
        with open(os.path.dirname(__file__) + "/config.yaml", "r") as f:
            custom_config = yaml.safe_load(f)
        return custom_config

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add sensor to the robot
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target((self.actions * self.cfg.action_scale).unsqueeze(-1), joint_ids=self._joint_dof_idx)

    def init_buffer(self):
        super_obs_buf = torch.zeros(
            self.num_envs,
            self.observation_space._shape[1],
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(self.obs_his_len):
            super_obs_buf[:, i * self.single_step_obs+1] = 1.0
        return super_obs_buf
    
    def reset_idx_buffer(self, env_ids: Sequence[int]):
        self.super_obs_buf[env_ids] = torch.zeros_like(self.super_obs_buf[env_ids])
        for i in range(self.obs_his_len):
            self.super_obs_buf[env_ids, i * self.single_step_obs+1] = 1.0
        self.last_actions[env_ids] = torch.zeros_like(self.last_actions[env_ids])

    def _get_observations(self) -> dict:
        self.obs = torch.cat((
            torch.ones(self.num_envs, 1, device=self.device)*self.velocity_input,
            self.robot.data.root_link_quat_w,
            self.robot.data.root_link_ang_vel_w,
            self.joint_pos[:, self._joint_dof_idx].squeeze(dim=2),
            self.joint_vel[:, self._joint_dof_idx].squeeze(dim=2),
            self.last_actions.clone().detach(),
        ), dim=-1)
        
        
        self.super_obs_buf[:, self.buffer_index*self.single_step_obs:(self.buffer_index+1)*self.single_step_obs] = self.obs
        
        self.buffer_index += 1
        if self.buffer_index >= self.obs_his_len:
            self.buffer_index = 0
        
        return {
            "policy": self.super_obs_buf.roll(-self.buffer_index*self.single_step_obs, dims=1),
            #! Below is for debugging
            # "root_obs": self.get_root_observations(),
            # "joint_obs": self.get_joint_observations(),
            # "contact_obs": self.get_contact_mask(),
        }
    
    def update_last_actions(self, actions):
        self.last_actions = torch.tensor(actions, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for i, vel in enumerate(self.robot.data.root_link_lin_vel_b):
            total_reward[i] += torch.exp(-(vel[0] - self.velocity_input)**2)*self.vx_lambda
            total_reward[i] += torch.exp(-vel[1]**2)*self.vy_lambda
            total_reward[i] += torch.exp(-vel[2]**2)*self.vz_lambda
            total_reward[i] += torch.exp(-self.robot.data.root_link_ang_vel_b[i, 2]**2)*self.wz_lambda
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        quats = self.robot.data.root_link_quat_w
        roll, pitch, _ = quat2euler(quats)
        angle_fail = (roll.abs() > math.pi / 4) | (pitch.abs() > math.pi / 4)
        z_low_fail = self.robot.data.root_pos_w[:, 2] < 0.6
        z_high_fail = self.robot.data.root_pos_w[:, 2] > 1.0
        z_fail = z_low_fail | z_high_fail
        
        out_of_bounds = z_fail | angle_fail

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.zeros_like(time_out, dtype=torch.bool)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        self.reset_idx_buffer(env_ids)
        
    
    def get_contact_mask(self) -> torch.Tensor:
        """Compute foot contact based on position and orientation."""
        l_foot_pos = self.robot.data.body_link_pos_w[:, self.l_roll_joint_idx, :]  # left foot
        r_foot_pos = self.robot.data.body_link_pos_w[:, self.r_roll_joint_idx, :]  # right foot
        l_foot_quat = self.robot.data.body_link_quat_w[:, self.l_roll_joint_idx, :]  # left foot
        r_foot_quat = self.robot.data.body_link_quat_w[:, self.r_roll_joint_idx, :]  # right foot
        contact_mask = torch.zeros(
            l_foot_pos.shape[0],
            self.sensor_relative_pos.shape[0] * 2, 
            dtype=torch.bool, 
            device=l_foot_pos.device)
        for i, [pos_l, pos_r, quat_l, quat_r] in enumerate(zip(l_foot_pos, r_foot_pos, l_foot_quat, r_foot_quat)):
            # Transform sensor positions to world coordinates
            transformed_sensor_pos_l = transform_points(self.sensor_relative_pos, quat_l[0], pos_l[0])
            transformed_sensor_pos_r = transform_points(self.sensor_relative_pos, quat_r[0], pos_r[0])
            # Compute contact mask based on height threshold
            contact_mask_l = transformed_sensor_pos_l[:, 2] < 0.01
            contact_mask_r = transformed_sensor_pos_r[:, 2] < 0.01
            contact_mask[i] = torch.cat((contact_mask_l, contact_mask_r), dim=0)
        return contact_mask

    def get_root_observations(self) -> torch.Tensor:
        """Compute root observations."""
        root_obs = torch.cat(
            (
                self.robot.data.root_link_pos_w,
                self.robot.data.root_link_quat_w,
            ),
            dim=-1,
        )
        return root_obs

    def get_joint_observations(self) -> torch.Tensor:
        """Compute joint observations."""
        joint_obs = torch.cat(
            (
                self.joint_pos[:, self._joint_dof_idx].unsqueeze(dim=1),  # joint positions
                self.joint_vel[:, self._joint_dof_idx].unsqueeze(dim=1),  # joint velocities
            ),
            dim=-1,
        )
        return joint_obs
    
    def get_pin_q(self) -> torch.Tensor:
        """Get q"""
        root_pos = self.robot.data.root_link_pos_w
        root_quat = self.robot.data.root_link_quat_w
        joint_pos = self.joint_pos[:, self._joint_dof_idx].squeeze(dim=2)

        mbc_obs = torch.zeros(
            root_pos.shape[0],
            root_pos.shape[1] + root_quat.shape[1] + joint_pos.shape[1],
            device=self.device,
        )
        for i in range(root_pos.shape[0]):
            mbc_obs[i, :3] = root_pos[i]
            mbc_obs[i, 3:7] = root_quat[i].roll(-1)
            mbc_obs[i, 7:] = joint_pos[i]
        return mbc_obs
    
    def get_pin_qvel(self) -> torch.Tensor:
        """Get qvel"""
        root_vel = self.robot.data.root_link_lin_vel_w
        root_ang_vel = self.robot.data.root_link_ang_vel_w
        joint_vel = self.joint_vel[:, self._joint_dof_idx].squeeze(dim=2)
        mbc_obs = torch.zeros(
            root_vel.shape[0],
            root_vel.shape[1] + root_ang_vel.shape[1] + joint_vel.shape[1],
            device=self.device,
        )
        for i in range(root_vel.shape[0]):
            mbc_obs[i, :3] = root_vel[i]
            mbc_obs[i, 3:6] = root_ang_vel[i].roll(-1)
            mbc_obs[i, 6:] = joint_vel[i]
        return mbc_obs


def quat2euler(quats: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to Euler angles."""
    w, x, y, z = quats.unbind(dim=-1)
    # Roll (x-axis rotation)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    # Pitch (y-axis rotation)
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    # Yaw (z-axis rotation)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw