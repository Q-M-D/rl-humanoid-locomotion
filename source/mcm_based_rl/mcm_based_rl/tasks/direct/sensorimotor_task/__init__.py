# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Sensorimotor-Task-Direct-v0",
    entry_point=f"{__name__}.sensorimotor_task_env:SensorimotorTaskEnv",
    disable_env_checker=True,
    kwargs={
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)