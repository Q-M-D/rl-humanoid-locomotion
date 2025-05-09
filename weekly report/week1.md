# Week 1 Progress Report

**Project Title**: Humanoid Robot Locomotion Control using Reinforcement Learning  
**Team Members**: Junlei Zhu, Jingwei Peng, Yansong Bai  
**Week**: 1 (May 5, 2025 – May 11, 2025)

---

## 1. Summary of Progress
- Set up the development environment successfully:
  - Installed required packages (IsaacGym, PyTorch, necessary RL libraries).
  - Verified simulation environment setup on local machines.
- Completed the initial implementation of the training code:
  - Script for PPO agent creation.
  - Script for environment interaction loop.
  - Logging and model checkpoint functionality.
- Verified that the code runs without critical errors (but not yet started actual training).

---

## 2. Challenges Encountered
- Minor compatibility issues between IsaacGym and the latest CUDA version, which were resolved by adjusting the environment configuration.
- Some initial confusion regarding the observation/action space definitions, which was clarified after reviewing IsaacGym documentation.

---

## 3. Plans for Next Week
- Start initial training runs with basic hyperparameters.
- Monitor training stability and reward curves.
- Fine-tune reward function based on early feedback.
- Begin implementing domain randomization techniques for robustness.

---

## 4. Current Risks / Concerns
- Training time may be longer than initially estimated; optimization of training efficiency may be necessary.
- Potential domain gap between simulation and real-world deployment needs further consideration.

