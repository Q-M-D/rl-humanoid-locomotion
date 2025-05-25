#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool

def contact_sensor_callback(data):
    # 处理接收到的接触传感器数据
    if data.data:
        rospy.loginfo("Contact detected!")
    else:
        rospy.loginfo("No contact detected.")

def contact_sensor_listener():
    # 初始化 ROS 节点
    rospy.init_node('contact_sensor_listener', anonymous=True)
    
    # 订阅接触传感器数据话题
    rospy.Subscriber("/contact_sensor_topic", Bool, contact_sensor_callback)
    
    # 保持节点运行，等待消息
    rospy.spin()

if __name__ == '__main__':
    try:
        contact_sensor_listener()
    except rospy.ROSInterruptException:
        pass
