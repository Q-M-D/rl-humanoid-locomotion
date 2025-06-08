# 📅 Week 5 Weekly Report

**Team**: Dora2 Locomotion Team  
**Week**: 5  
**Period**: 2025.06.03 – 2025.06.07  
**Project**: Sim-to-Real Humanoid Locomotion with PPO and Estimator-Augmented Policy

---

## ✅ 1. Progress Summary

This week marks the final phase of our project, where we successfully transitioned from simulation to physical deployment of our learned locomotion policy. Key milestones include:

- ✅ **Sim-to-Real Deployment Completed**  
  Deployed the PPO-trained policy onto the physical Dora2 bipedal humanoid robot. The robot demonstrates stable omnidirectional walking, accurate velocity tracking via joystick commands, and robust performance under minor disturbances.

- ✅ **Observer Module Integrated**  
  Integrated a state estimator that infers contact masks and floating base velocity from historical proprioception, improving stability in real deployment.

- ✅ **Final Report Completed**  
  Compiled all components of the project—including algorithm design, reward engineering, sim-to-real methodology, and experiments—into a LaTeX-based final report.

- ✅ **Poster Designed and Delivered**  
  Designed a presentation poster summarizing our methodology, results, and key insights, including training curves and deployment images.

---

## 📈 2. Achievements Compared to Goals

| Objective                                       | Status      | Notes                                                                 |
|------------------------------------------------|-------------|-----------------------------------------------------------------------|
| Simulated PPO training with domain randomization | ✅ Completed | Trained in Isaac Gym, validated in MuJoCo                            |
| Policy generalization across simulators         | ✅ Verified  | Successfully transferred from Isaac Gym to MuJoCo                    |
| Observer-augmented policy                       | ✅ Integrated| Improved base velocity and contact estimation                        |
| Real-world deployment on Dora2                  | ✅ Completed | Stable walking, low fall rate, joystick-controllable locomotion      |
| Final report and poster                         | ✅ Delivered | All presentation materials finalized this week                       |

---

## 📌 3. Key Insights

- Asymmetric PPO (with privileged critic inputs) helps with more stable training and better sim-to-real transfer.
- Domain randomization (mass, friction, sensor noise) is crucial for closing the sim-to-real gap.
- Observer improves real-world stability by estimating non-directly observable states.

---

## 🧠 4. Challenges Encountered

- Minor discrepancies in base acceleration between Isaac Gym and Dora2 hardware led to occasional instability.
- Real-world joystick control introduced unexpected latency, requiring modeling in training.

