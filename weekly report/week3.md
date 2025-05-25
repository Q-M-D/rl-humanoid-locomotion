# Week 3 Progress Report

**Project Title**: Humanoid Robot Locomotion Control using Reinforcement Learning  
**Team Members**: Junlei Zhu, Jingwei Peng, Yansong Bai  
**Week**: 3 (May 19, 2025 – May 25, 2025)

---

## 1. Summary of Progress
- Implemented turning/steering functionality in the trained locomotion model.  
- Verified successful export of the model from PyTorch (PT) to ONNX format, ensuring compatibility with C++ inference pipelines.  
- Validated that when no velocity commands are given, the robot can maintain a stable idle state without drifting or collapsing.

---

## 2. Challenges Encountered
- Ensuring consistency of model outputs after exporting to ONNX and running on C++ required careful operator compatibility checks.  
- The steering behavior needed fine-tuning to avoid over-rotation and to ensure stability during direction changes.  
- Gazebo simulation results were not ideal; the model exhibited instability or undesired behaviors when tested in the physics-rich environment.

---

## 3. Plans for Next Week
1. Improve the robustness of the control model, especially under dynamics mismatch and noise.  
2. Investigate and enhance performance within the Gazebo simulator.  
3. Analyze potential discrepancies between Isaac Gym and Gazebo to improve sim-to-sim transfer fidelity.

---

## 4. Current Risks / Concerns
- ONNX model fidelity: slight deviations between PyTorch and ONNX may accumulate during deployment.  
- Gazebo physics and sensor noise may expose weaknesses in the trained policy, requiring domain adaptation.  
- Steering behavior under variable speed commands may require further training or reward adjustment.

---

## 5. Contributions

| Team Member | Contributions                                                        |
|:------------|:-----------------------------------------------------------------------|
| Junlei Zhu  | Steering logic implementation, ONNX export and C++ integration         |
| Jingwei Peng| Testing and validating model behavior in idle and turning scenarios    |
| Yansong Bai | Preliminary Gazebo integration, robustness evaluation under noise      |

---
