# Week 4 Progress Report

**Project Title**: Humanoid Robot Locomotion Control using Reinforcement Learning  
**Team Members**: Junlei Zhu, Jingwei Peng, Yansong Bai  
**Week**: 4 (May 26, 2025 – June 1, 2025)

---

## 1. Summary of Progress
- Successfully ran the entire control pipeline end-to-end, from training to deployment.  
- Deployed the model on the real robot hardware for the first time.  
- Verified that the system can boot, execute inference, and actuate the robot in real-time.  
- Observed sim-to-real issues: the robot collapsed after a few steps, likely due to insufficient robustness and domain randomization.

---

## 2. Challenges Encountered
- **Real-world instability**: The deployed policy could not maintain walking on hardware due to the sim-to-real gap, suggesting the need for more extensive domain randomization during training.  
- **Hardware damage**: During real-world experiments, the robot suffered physical damage, halting further testing. Emergency repairs were conducted over the weekend to restore functionality.  
- Limited time for real-world iterations due to hardware downtime.

---

## 3. Plans for Next Week
1. Enhance simulation training with more domain randomization (e.g., latency, noise, friction variations).  
2. Re-train and fine-tune the model with added robustness constraints.  
3. Resume real-world experiments once the robot is repaired and functional.

---

## 4. Current Risks / Concerns
- Hardware fragility and time loss due to repair work reduce the iteration cycle for real-world deployment.  
- Without sufficient domain randomization, retrained policies may continue to fail upon deployment.  
- Robot safety and reliability during repeated real-world trials are a significant concern.

---

## 5. Contributions

| Team Member | Contributions                                                         |
|:------------|:----------------------------------------------------------------------|
| Junlei Zhu  | Real-world deployment pipeline, debugging sim-to-real issues          |
| Jingwei Peng| Hardware setup, execution of real-world testing, initial failure analysis |
| Yansong Bai | Emergency diagnosis and coordination of robot repair                 |

---
