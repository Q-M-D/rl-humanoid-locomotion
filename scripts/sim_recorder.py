import mujoco
import numpy as np
import json
from rich import print

class SimRecorder:
    def __init__(self, m, d):
        self.m = m
        self.d = d
        
        self.qpos = []    # don't contain the floating base 
        self.qvel = []    # don't contain the floating base
        self.fb_orientation = [] # quaternion for the floating base orientation, (cos(a/2), sin(a/2)*(x,y,z)) => (w,x,y,z)
        self.fb_angular_velocity = [] # angular velocity for the floating base
        self.control_input = []   # position or torque
        self.contact_mask = []   # contact info
        self.time = []  # time
        self.vel_command = []
        
        self.recorder = {
            "qpos": self.qpos,
            "qvel": self.qvel,
            "fb_orientation": self.fb_orientation,
            "fb_angular_velocity": self.fb_angular_velocity,
            "control_input": self.control_input,
            "contact_mask": self.contact_mask,
            "time": self.time,
            "vel_command": self.vel_command
        }
        
    def update(self, concou):
        self.recorder["qpos"].append(self.d.qpos[7:].copy())
        self.recorder["qvel"].append(self.d.qvel[6:].copy())
        self.recorder["fb_orientation"].append(self.d.qpos[3:7].copy())
        self.recorder["fb_angular_velocity"].append(self.d.qvel[3:6].copy())
        self.recorder["control_input"].append(self.d.ctrl.copy())
        self.recorder["contact_mask"].append(concou)
        self.recorder["time"].append(self.d.time)
        
    def get_recorder(self):
        return self.recorder
    
    def record_vel_command(self, final_pos, end_time):
        """
        Record the velocity command for the simulation
        ---
        input: `final_pos` - final position of the floating base
               `end_time` - end time of the simulation
        """
        vel_command = final_pos / end_time
        self.recorder["vel_command"] = [vel_command] * len(self.recorder["qpos"])
        print("Velocity command recorded:", vel_command)
    
    def save_to_json_file(self, filename):
        data_to_save = {}
        for key, value in self.recorder.items():
            data_to_save[key] = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in value]
        
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"Data saved to {filename}")
        print(f"Data length: {len(data_to_save['qpos'])}")
