# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from typing import Sequence
import yaml

from omniisaacgymenvs.utils.terrain_utils import spawn_ground_plane
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import GroundPlaneCfg
import isaaclab.sim as sim_utils

from .sensorimotor_task_env_cfg import SensorimotorTaskEnvCfg


class SensorimotorTaskEnv(DirectRLEnv):
    cfg: SensorimotorTaskEnvCfg

    def __init__(self, cfg: SensorimotorTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # 获取机器人属性设置
        self._joint_dof_idx = self.robot.find_joints(self.cfg.controllable_dof_names)
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        # 加载自定义配置
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
        
        # 初始化观察缓冲区
        self.super_obs_buf = self.init_buffer()
        self.buffer_index = 0
        self.last_actions = torch.zeros(self.num_envs, self.actions_dim, device=self.device)
        
        # 创建固定大小的观察缓冲区
        self.obs_buffer_size = 20
        self.obs_buffer = torch.zeros(
            self.obs_buffer_size,
            self.observation_space._shape[1],
            dtype=torch.float32,
            device=self.device,
        )
        self.obs_buffer_index = 0  # 跟踪缓冲区的当前位置

        self.obs = None

    def custom_config(self):
        # 从配置文件加载自定义设置
        with open(os.path.dirname(__file__) + "/config.yaml", "r") as f:
            custom_config = yaml.safe_load(f)
        return custom_config

    def _setup_scene(self):
        # 设置场景
        self.robot = Articulation(self.cfg.robot_cfg)
        # 添加地面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # 克隆并复制环境
        self.scene.clone_environments(copy_from_source=False)
        # 将机器人添加到场景
        self.scene.articulations["robot"] = self.robot
        
        # 添加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 物理模拟步骤前的准备
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # 应用动作到机器人
        self.robot.set_joint_position_target((self.actions * self.cfg.action_scale).unsqueeze(-1), joint_ids=self._joint_dof_idx)

    def init_buffer(self):
        # 初始化观察缓冲区
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
        # 重置指定环境的缓冲区
        self.super_obs_buf[env_ids] = torch.zeros_like(self.super_obs_buf[env_ids])
        for i in range(self.obs_his_len):
            self.super_obs_buf[env_ids, i * self.single_step_obs+1] = 1.0
        self.last_actions[env_ids] = torch.zeros_like(self.last_actions[env_ids])

    def update_observation_buffer(self, new_observation: torch.Tensor) -> None:
        """
        更新观察缓冲区，实现先进先出机制
        
        Parameters
        ----------
        new_observation : torch.Tensor
            要添加到缓冲区的最新观察
        """
        # 将新观察添加到缓冲区当前索引位置
        self.obs_buffer[self.obs_buffer_index] = new_observation
        
        # 更新缓冲区索引（循环）
        self.obs_buffer_index = (self.obs_buffer_index + 1) % self.obs_buffer_size

    def get_observation_buffer(self) -> torch.Tensor:
        """
        获取观察缓冲区的当前状态
        
        Returns
        ----------
        torch.Tensor
            包含最后 buffer_size 个观察的缓冲区
        """
        return self.obs_buffer

    def _get_observations(self) -> dict:
        # 获取观察数据
        # 1. command velocity
        # 2. feedback orientation
        fb_orientation = self.robot.data.root_quat_w
        # 3. feedback angular velocity
        fb_angular_velocity = self.robot.data.root_angular_vel.flatten()
        # 4. joint position
        self.joint_pos = self.robot.data.joint_pos[:, self._joint_dof_idx].squeeze(dim=2)
        # 5. joint velocity
        self.joint_vel = self.robot.data.joint_vel[:, self._joint_dof_idx].squeeze(dim=2)
        # 6. control input
        control_input = self.robot.data.joint_effort_target[:, self._joint_dof_idx].squeeze(dim=2)
        
        # 滚动缓冲区并添加新的观察
        self.super_obs_buf = torch.roll(self.super_obs_buf, -self.single_step_obs, dims=1)
        
        # 将新的观察添加到缓冲区的末尾
        self.super_obs_buf[:, -self.single_step_obs] = torch.cat(
            [
                self.velocity_input.expand(self.num_envs, 1),
                fb_orientation,
                fb_angular_velocity,
                self.joint_pos,
                self.joint_vel,
                control_input,
            ],
            dim=1,
        )
        
        # 更新观察缓冲区
        self.update_observation_buffer(self.super_obs_buf[0].clone())  # 只记录第一个环境的观察
        
        self.obs = self.super_obs_buf.clone()
        return {
            "policy": self.obs,
        }

    def update_last_actions(self, actions):
        # 更新上一步的动作
        self.last_actions = actions.clone()

    def _get_rewards(self) -> torch.Tensor:
        # 计算奖励
        # 获取速度信息
        velocity = self.robot.data.root_lin_vel
        vx = velocity[:, 0]
        vy = velocity[:, 1]
        vz = velocity[:, 2]
        
        # 获取角速度信息
        angular_velocity = self.robot.data.root_angular_vel
        wz = angular_velocity[:, 2]
        
        # 根据速度和角速度计算奖励
        reward = self.vx_lambda * vx - \
                 self.vy_lambda * torch.abs(vy) - \
                 self.vz_lambda * torch.abs(vz) - \
                 self.wz_lambda * torch.abs(wz)
                 
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 判断回合是否结束的条件
        # 获取机器人高度
        height = self.robot.data.root_pos[:, 2]
        # 获取机器人姿态
        quaternion = self.robot.data.root_quat_w
        
        # 根据高度和姿态设置结束条件
        reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        reset = torch.where(height < 0.4, torch.ones_like(reset), reset)
        reset = torch.where(quaternion[:, 0] < 0.0, torch.ones_like(reset), reset)
        
        # 重置信号和终止信号
        return reset, reset

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # 重置指定环境的状态
        if env_ids is not None and len(env_ids) > 0:
            # 重置机器人状态
            self.robot.initialize(env_ids=env_ids)
            # 重置缓冲区
            self.reset_idx_buffer(env_ids)