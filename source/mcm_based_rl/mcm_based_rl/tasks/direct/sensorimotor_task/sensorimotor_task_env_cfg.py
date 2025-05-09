# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from gymnasium.spaces import Box
import numpy as np

PROJECT_DIR = "/home/mmlab-rl/codes/sensorimotor-rl"

DORA2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PROJECT_DIR}/model/usd/dora2_stand/dora2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_pitch_joint": 0.0,
            ".*_knee_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            ".*_ankle_roll_joint": 0.0,
        },
    ),
    actuators={
        "hip_actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*"],
            effort_limit=400.0,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_roll_joint": 80.0,
                ".*_yaw_joint": 80.0,
                ".*_pitch_joint": 80.0,
            },
            damping={
                ".*_roll_joint": 5.0,
                ".*_yaw_joint": 4.0,
                ".*_pitch_joint": 4.0,
            },
        ),
        "knee_actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=400.0,
            velocity_limit_sim=100.0,
            stiffness=50.0,
            damping=4.0
        ),
        "ankle_actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*"],
            effort_limit=400.0,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_pitch_joint": 30.0,  # ankle pitch
                ".*_roll_joint": 30.0,   # ankle roll
            },
            damping={
                ".*_pitch_joint": 0.5,    # ankle pitch
                ".*_roll_joint": 0.5,     # ankle roll
            },
        ),
    },
)
"""Configuration for the Dora2 robot using force control."""


@configclass
class SensorimotorTaskEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2  # 控制步骤频率
    episode_length_s = 10.0  # 每个回合的长度（秒）
    
    # - spaces definition
    action_space = Box(
        low=-5.0,
        high=5.0,
        shape=(12,),
        dtype=np.float32
    )
    
    """
    Observation space dimension:
    1: command velocity
    4: feedback orientation
    3: feedback angular velocity
    12: joint position
    12: joint velocity
    12: control input
    
    Total per step: 44
    Buffer size: 5 steps
    Final dimension: 44 * 5 = 220
    """
    observation_space = (1+4+3+12+12+12)*5
    
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 500, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = DORA2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    controllable_dof_names = [
        'l_leg_hip_roll_joint', 'l_leg_hip_yaw_joint',      'l_leg_hip_pitch_joint', 'l_leg_knee_joint',     'l_leg_ankle_pitch_joint',  'l_leg_ankle_roll_joint',
        'r_leg_hip_roll_joint', 'r_leg_hip_yaw_joint',      'r_leg_hip_pitch_joint', 'r_leg_knee_joint',     'r_leg_ankle_pitch_joint',  'r_leg_ankle_roll_joint'
    ]
    # - action scale
    action_scale = 1.0