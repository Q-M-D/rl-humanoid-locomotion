# Contact sensor positions

Relative to the joint `*_leg_ankle_roll_joint`, it is:

```python
sensor_pos = [[0.1, 0.038, -0.05],
    [0.03, 0.042, -0.05],
    [-0.02, 0.04, -0.05],
    [-0.05, 0.03, -0.05],
    [0.12, 0.0, -0.05],
    [0.052, 0.0, -0.05],
    [0.009, 0.0, -0.05],
    [-0.065, 0.0, -0.05],
    [0.1, -0.038, -0.05],
    [0.03, -0.042, -0.05],
    [-0.02, -0.04, -0.05],
    [-0.05, -0.03, -0.05],]
```

# Stiffness and damping

  <!-- control:
    stiffness:
      leg_l1_joint: 90
      leg_l2_joint: 80
      leg_l3_joint: 80
      leg_l4_joint: 80
      leg_l5_joint: 10
      leg_l6_joint: 10
  
      leg_r1_joint: 90
      leg_r2_joint: 80
      leg_r3_joint: 80
      leg_r4_joint: 80
      leg_r5_joint: 10
      leg_r6_joint: 10
    damping:
      leg_l1_joint: 5.0
      leg_l2_joint: 4.0
      leg_l3_joint: 4.0
      leg_l4_joint: 4.0
      leg_l5_joint: 0.5
      leg_l6_joint: 0.5
  
      leg_r1_joint: 5.0
      leg_r2_joint: 4.0
      leg_r3_joint: 4.0
      leg_r4_joint: 4.0
      leg_r5_joint: 0.5
      leg_r6_joint: 0.5 -->

```python
stiffness = {
    'l_leg_hip_roll_joint': 90,
    'l_leg_hip_yaw_joint': 80,
    'l_leg_hip_pitch_joint': 80,
    'l_leg_knee_joint': 80,
    'l_leg_ankle_pitch_joint': 10,
    'l_leg_ankle_roll_joint': 10,
    'r_leg_hip_roll_joint': 90,
    'r_leg_hip_yaw_joint': 80,
    'r_leg_hip_pitch_joint': 80,
    'r_leg_knee_joint': 80,
    'r_leg_ankle_pitch_joint': 10,
    'r_leg_ankle_roll_joint': 10,
}

damping = {
    'l_leg_hip_roll_joint': 5.0,
    'l_leg_hip_yaw_joint': 4.0,
    'l_leg_hip_pitch_joint': 4.0,
    'l_leg_knee_joint': 4.0,
    'l_leg_ankle_pitch_joint': 0.5,
    'l_leg_ankle_roll_joint': 0.5,
    'r_leg_hip_roll_joint': 5.0,
    'r_leg_hip_yaw_joint': 4.0,
    'r_leg_hip_pitch_joint': 4.0,
    'r_leg_knee_joint': 4.0,
    'r_leg_ankle_pitch_joint': 0.5,
    'r_leg_ankle_roll_joint': 0.5,
}
```

# Joints range

```python
joint_range = {
    'l_leg_hip_roll_joint': [-0.3, 1],
    'l_leg_hip_yaw_joint': [-0.75, 1.05],
    'l_leg_hip_pitch_joint': [-1.32, 0.68],
    'l_leg_knee_joint': [0.0, 2.01],
    'l_leg_ankle_pitch_joint': [-0.88, 0.68],
    'l_leg_ankle_roll_joint': [-0.5, 0.5],
    'r_leg_hip_roll_joint': [-0.3, 1],
    'r_leg_hip_yaw_joint': [-0.75, 1.05],
    'r_leg_hip_pitch_joint': [-1.32, 0.68],
    'r_leg_knee_joint': [0.0, 2.01],
    'r_leg_ankle_pitch_joint': [-0.88, 0.68],
    'r_leg_ankle_roll_joint': [-0.5, 0.5],
}
```