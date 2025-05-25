#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32

def talker():
    # 创建一个发布者，发布到 'signal_topic' 话题，消息类型为 Int32
    pub = rospy.Publisher('signal_topic', Int32, queue_size=10)
    
    # 初始化节点，节点名称为 'python_talker_node'
    rospy.init_node('python_node', anonymous=True)
    
    # 设置循环频率为 10Hz
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        signal = 1  # 输出信号 1
        
        # 输出日志信息
        rospy.loginfo(f"Publishing signal: {signal}")
        
        # 发布消息
        pub.publish(signal)
        
        # 等待一段时间
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
