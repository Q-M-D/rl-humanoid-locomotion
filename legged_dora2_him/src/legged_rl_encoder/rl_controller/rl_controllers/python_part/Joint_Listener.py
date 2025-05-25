import rospy
from std_msgs.msg import Float64MultiArray

def Joint_pos_callback(data):
    # 处理接收到的接触传感器数据
    '''
    接受到的信息为：
    {"leg_l1_joint", jointPos(0)}, {"leg_l2_joint", jointPos(1)}, {"leg_l3_joint", jointPos(2)}, 
    {"leg_l4_joint", jointPos(3)}, {"leg_l5_joint", jointPos(4)}, {"leg_l6_joint", jointPos(5)}, 
    {"leg_r1_joint", jointPos(6)}, {"leg_r2_joint", jointPos(7)}, {"leg_r3_joint", jointPos(8)}, 
    {"leg_r4_joint", jointPos(9)}, {"leg_r5_joint", jointPos(10)}, {"leg_r6_joint", jointPos(11)}};
    
    对应到控制算法中的：
    'l_leg_hip_roll_joint':     model.getJointId('l_leg_hip_roll_joint'),
    'l_leg_hip_yaw_joint':      model.getJointId('l_leg_hip_yaw_joint'),
    'l_leg_hip_pitch_joint':    model.getJointId('l_leg_hip_pitch_joint'),
    'l_leg_knee_joint':         model.getJointId('l_leg_knee_joint'),
    'l_leg_ankle_pitch_joint':  model.getJointId('l_leg_ankle_pitch_joint'),
    'l_leg_ankle_roll_joint':   model.getJointId('l_leg_ankle_roll_joint'),
    'r_leg_hip_roll_joint':     model.getJointId('r_leg_hip_roll_joint'),
    'r_leg_hip_yaw_joint':      model.getJointId('r_leg_hip_yaw_joint'),
    'r_leg_hip_pitch_joint':    model.getJointId('r_leg_hip_pitch_joint'),
    'r_leg_knee_joint':         model.getJointId('r_leg_knee_joint'),
    'r_leg_ankle_pitch_joint':  model.getJointId('r_leg_ankle_pitch_joint'),
    'r_leg_ankle_roll_joint':   model.getJointId('r_leg_ankle_roll_joint'),
    
    
    缺少力控输入接口[是否需要自行写]
    模型需要加入与地面的接触点信息，同时向外部传出接触信息
        1) urdf中要加入接触点的sensor
        2) 需要对sensor进行重写,同时需要编写新的节点传输sensor信息
        
    
    '''
    if data.data:
        rospy.loginfo("data_msg:" + str(data) + "\n")
        
    else:
        rospy.loginfo("No info detected.")

def contact_sensor_listener():
    # 初始化 ROS 节点
    rospy.init_node('Joint_Listener', anonymous=True)
    
    # 订阅接触传感器数据话题
    rospy.Subscriber("/data_analysis/real_joint_pos", Float64MultiArray, Joint_pos_callback)
    
    # 保持节点运行，等待消息
    rospy.spin()

if __name__ == '__main__':
    try:
        contact_sensor_listener()
    except rospy.ROSInterruptException:
        pass