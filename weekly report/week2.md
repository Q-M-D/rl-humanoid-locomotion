# Week 2 Progress Report

**Project Title**: Humanoid Robot Locomotion Control using Reinforcement Learning  
**Team Members**: Junlei Zhu, Jingwei Peng, Yansong Bai  
**Week**: 2 (May 12, 2025 – May 18, 2025)

---

## 1. Summary of Progress
- Completed robot locomotion implementation in the Isaac Gym simulation environment, including gait generation, joint control, and balance maintenance.  
- Integrated gamepad control (e.g., Xbox controller) for interactive/manual driving of the robot.  
- Designed and implemented a comprehensive reward engine consisting of 24 distinct reward components to guide the PPO training process.

---

## 2. Challenges Encountered
- Tuning multiple reward components to produce smooth and stable walking behavior required extensive testing and adjustments.  
- Ensuring the gamepad control inputs mapped intuitively to robot actions demanded refinement of input scaling and deadzone configurations.  
- Debugging edge cases in simulation (e.g., foot clipping or unintended contacts) exposed subtle issues in the contact-handling reward terms.

---

## 3. Reward Function Overview

| Reward Name               | Description                                                                                                         |
|---------------------------|---------------------------------------------------------------------------------------------------------------------|
| `joint_pos`               | Reward for closeness between current joint positions and target joint positions                                     |
| `feet_distance`           | Penalizes feet being too close or too far apart                                                                      |
| `knee_distance`           | Rewards maintaining a proper knee separation                                                                         |
| `foot_slip`               | Penalizes horizontal slipping of feet in contact                                                                     |
| `feet_air_time`           | Rewards longer foot air time to encourage larger step lengths                                                        |
| `feet_contact_number`     | Rewards correct number of foot contacts according to gait phase                                                      |
| `orientation`             | Rewards keeping the robot’s base orientation level                                                                  |
| `feet_contact_forces`     | Penalizes excessive contact forces on feet                                                                           |
| `default_joint_pos`       | Rewards joints close to default positions (focus on yaw and roll axes)                                              |
| `base_height`             | Rewards base height remaining within desired range                                                                  |
| `base_acc`                | Penalizes high base acceleration to encourage smooth motion                                                         |
| `vel_mismatch_exp`        | Rewards small mismatches in linear and angular velocities                                                           |
| `track_vel_hard`          | Rewards accurate tracking of commanded linear and angular velocities                                                |
| `tracking_lin_vel`        | Rewards tracking of linear velocity commands                                                                        |
| `tracking_ang_vel`        | Rewards tracking of angular (yaw) velocity commands                                                                 |
| `feet_clearance`          | Rewards appropriate foot clearance during swing phase                                                               |
| `low_speed`               | Rewards maintaining speed within desired range and penalizes direction mismatches                                   |
| `torques`                 | Penalizes high joint torques to encourage energy-efficient movement                                                 |
| `dof_vel`                 | Penalizes high joint velocities                                                                                     |
| `dof_acc`                 | Penalizes high joint accelerations                                                                                  |
| `collision`               | Penalizes collisions between designated body parts and environment                                                  |
| `action_smoothness`       | Penalizes abrupt changes in actions to produce smoother motion                                                      |

---

## 4. Current Risks / Concerns
- The large number of reward components may introduce conflicting objectives, potentially slowing convergence.  
- Gamepad control validation may not fully cover edge cases encountered during autonomous training.  
- Training time could become a bottleneck; consider parallelizing environments or reducing model complexity if needed.

---
## 5. Plans for Next Week
1. Fix bugs: address steering/turning issues.  
2. Begin real-robot deployment tests on Dora2.  
3. Develop and integrate a contact predictor module to better handle contact information.

---
## 6. Contributions

| Team Member | Contributions                                              |
|:------------|:-----------------------------------------------------------|
| Junlei Zhu  | Isaac Gym locomotion setup, reward engine core functions   |
| Jingwei Peng| Gamepad control integration, simulation debugging          |
| Yansong Bai | Reward weight tuning, logging and evaluation framework     |

---
