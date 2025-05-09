import mujoco
import mujoco.viewer
import time
import numpy as np
from mbc import MBC, walk_controller
import os
from rich import print
import yaml
from sim_recorder import SimRecorder
import matplotlib.pyplot as plt


# Initialize mujoco
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

mjcf_folder = config['mjcf_folder']

modelfilename = os.path.join(mjcf_folder, "dora2_stand_fix.xml")
modelfilename_pin = os.path.join(mjcf_folder, "dora2_stand_fix_pin.xml")

m = mujoco.MjModel.from_xml_path(modelfilename)
d = mujoco.MjData(m)
c1 = MBC(modelfilename_pin)
simRecorder = SimRecorder(m, d)

end_time = 50


def set_cam_follow(m, viewer, body_name="base_link"):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    viewer.cam.distance = 4.0  # 设置摄像机距离
    viewer.cam.elevation = -30  # 设置摄像机仰角
    viewer.cam.azimuth = 135  # 设置摄像机方位角


def set_cam_fix(m, viewer):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.lookat = [0, 0, 0]
    viewer.cam.distance = 5.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 45


def get_sensor_position(m, d, sensor_list):
    """
    Get sensor position in absolute axes
    ---
    input: `sensor_list` - select sensors's name list

    output: position in the world coordinate system
    """
    return d.site_xpos[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sensor) for sensor in sensor_list], 2]


def get_sensor_info(m, d, c1, method="force"):
    """
    Get sensor information
    ---
    input: `method` - method to get sensor information, "position" or "force"
    ---
    output: sensor contact information
    """
    concou = np.zeros(len(c1.fclist))
    if method == "position":
        sensor_height = get_sensor_position(m, d, c1.fclist)
        for i in range(len(c1.fclist)):
            if sensor_height[i] < 0.01:
                concou[i] += 1
    elif method == "force":
        concou += d.sensordata[:len(c1.fclist)]
    return concou


def get_pos(d):
    """
    Get position in the world coordinate system
    ---
    input: `d` - mujoco data

    output: position in the world coordinate system
    """
    qpos = d.qpos.copy()
    qpos[3:7] = np.roll(qpos[3:7], -1)  # roll the quaternion
    return qpos


mvcomhist = []
with mujoco.viewer.launch_passive(m, d) as viewer:
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1


    set_cam_follow(m, viewer, body_name="base_link")
    start = time.time()
    mujoco.mj_forward(m, d)
    while viewer.is_running() and time.time() - start < end_time:
        step_start = time.time()
        
        ppos = get_pos(d)

        concou = get_sensor_info(m, d, c1)
        c1.centerofhip = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'hip_center')]
        mujoco.mj_subtreeVel(m,d)
        c1.mvcom = d.subtree_linvel[1].copy()
        mvcomhist.append(c1.mvcom[0])
        c1.lramp = 0.2          # 左右摆动幅度
        c1.kneeliftangle = 1.0  # 膝盖抬升角度
        c1.amoo = 0.08          # 步态振幅参数
        c1.anklebias =  -0.03   #脚踝的上仰角度
        c1.b = 0.2              #步子相对于速度的敏感度
        
        # # 向前走1
        c1.anklebias =  0.09
        
        # 向前走2，快一点
        # c1.anklebias =  0.14
        
        # # 向前走3，快一点
        # c1.anklebias =  0.2
        # c1.lramp = 0.19
        
        # # 左转
        # c1.anklebias =  0.09
        # c1.hipzl = -0.2
        
        # # 右转
        # c1.anklebias =  0.09
        # c1.hipzr = 0.2
        
        # 向后
        # c1.anklebias =  -0.03

        d.ctrl[:] = walk_controller(c1, ppos, d.qvel, concou, time=d.time)
        simRecorder.update(concou)
        mujoco.mj_step(m, d)

        viewer.sync()
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

final_pos = d.qpos[0].copy()
simRecorder.record_vel_command(final_pos, d.time)


script_dir = os.path.dirname(os.path.abspath(__file__))
simRecorder.save_to_json_file(os.path.join(script_dir, "forward_fast.json"))
# print(mvcomhist)
def mv_average(data, window_size):
    kernel = np.ones(window_size) / window_size
    data_array = np.array(data)
    return np.convolve(data_array, kernel, mode='same')
#plt.plot(mvcomhist)
smooth_mvcomhist = mv_average(mvcomhist, 130)
plt.plot(smooth_mvcomhist)
# plt.show()
plt.savefig("output.png")
# np.average(mvcomhist[])
