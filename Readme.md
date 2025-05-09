# Project Plan Document

## Project Title
**Humanoid Robot Locomotion Control using Reinforcement Learning**

Authors: Junlei Zhu, Jingwei Peng, Yansong Bai  
Emails: {zhujl12024, pengjw2024, baiys2022}@shanghaitech.edu.cn  
Date: May 2025

---

## Project Overview
Humanoid robots hold great potential for real-world applications. However, their locomotion control remains a complex and unsolved challenge.  
This project proposes the use of **Proximal Policy Optimization (PPO)**, a state-of-the-art reinforcement learning algorithm, to address the motion control issues inherent in humanoid robotics.  

By leveraging PPO, we aim to develop a robust framework capable of learning adaptable control strategies that manage the intricate balance and coordination required for humanoid locomotion and manipulation.  
The project will focus on training and deploying a walking controller for the **Dora2** humanoid robot, first in simulation and subsequently on real hardware.

---

## Objectives
- Implement a PPO-based locomotion controller for humanoid robots.
- Achieve stable and efficient autonomous walking in simulation.
- Successfully transfer the learned control policy from simulation to a real-world Dora2 robot.

---

## Key Features
- **Reinforcement Learning Control**: Training robust policies for walking using PPO.
- **Domain Randomization**: Improve transferability from simulation to real-world deployment.
- **Safety-Conscious Deployment**: Enforce joint limits and safety stops during real-world experiments.

---

## Technology Stack
- Programming Languages: Python
- Simulation Environment: IsaacGym / IsaacLab
- RL Algorithms: PPO (Proximal Policy Optimization)
- Robot Platform: Dora2 humanoid robot
- Frameworks/Tools: PyTorch, Omniverse Isaac Sim (optional for visualization)

---

## Methodology

### Problem Formulation
- We model humanoid walking as a **Partially Observable Markov Decision Process (POMDP)**.
- The robot acts based on partial observations (e.g., noisy sensors) and selects torque actions.
- The environment satisfies the Markov property in simulation.

### Reinforcement Learning Algorithm: PPO
- PPO optimizes a clipped surrogate objective to ensure stable policy updates.
- We parameterize the policy and value networks using neural networks.

### Reward Design
The reward function components include:
- **Forward Progress**: Reward proportional to distance moved forward.
- **Energy Efficiency**: Penalty for excessive torque and joint velocity.
- **Stability**: Penalty for large center of mass deviations or falls.
- **Smoothness**: Penalty for abrupt joint movements.

### Domain Randomization
Randomize physical parameters, sensor noise, and external disturbances during training to enhance robustness.

### Simulation Environment
- Use IsaacGym or IsaacLab for large-scale parallel simulation.
- Observations: joint states, IMU data.
- Actions: target joint torques.

### Real Robot Deployment
- Policies are transferred to the Dora2 robot.
- Real-time safety checks and visualization are incorporated during deployment.

---

## Expected Goals
- **Simulation Success**: Achieve stable walking behavior in simulation with minimal falls and smooth gait patterns.
- **Real-World Transfer**: Deploy trained policy on Dora2 robot with minimal performance loss and successful balance maintenance under disturbances.

---

