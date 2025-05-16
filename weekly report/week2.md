# Week 2 Progress Report

**Project Title**: Humanoid Robot Locomotion Control using Reinforcement Learning  
**Team Members**: Junlei Zhu, Jingwei Peng, Yansong Bai  
**Week**: 2 (May 12, 2025 – May 18, 2025)

---

## 1. Summary of Progress
- Completed the initial implementation of the reward engine:
  - Defined and implemented key reward components:
    - **Forward Progress**: Rewards movement in the desired walking direction.
    - **Energy Efficiency**: Penalizes high torque and joint velocities.
    - **Stability**: Penalizes deviation of center of mass and falling.
    - **Smoothness**: Penalizes abrupt joint motion changes.
  - Integrated the reward engine into the training pipeline.
  - Verified reward values through simulation rollouts and logging.
- Refactored parts of the training loop to better accommodate modular reward design.

---

## 2. Challenges Encountered
- Designing reward weights to balance learning objectives required multiple iterations.
- Ensuring reward signals were neither too sparse nor too noisy during testing.
- Debugging reward behavior in edge cases (e.g., robot falling immediately on reset).

---

## 3. Plans for Next Week
- Begin full training runs using the current PPO implementation and reward engine.
- Monitor training stability and learning curves.
- Visualize agent behavior and assess qualitative locomotion results.
- Start preparing domain randomization code modules.

---

## 4. Current Risks / Concerns
- Reward shaping still needs tuning to ensure convergence and meaningful gait.
- Training speed may become a bottleneck once full training begins.
- Robustness to sensor noise and external disturbances still untested.

---

## 5. Contributions
| Team Member | Contributions |
| :---------- | :------------- |
| Junlei Zhu  | Reward module design and implementation |
| Jingwei Peng| Reward integration with PPO pipeline |
| Yansong Bai | Testing and logging of reward values, feedback on shaping strategy |

---
