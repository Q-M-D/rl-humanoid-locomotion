/********************************************************************************
Modified Copyright (c) 2023-2024, BridgeDP Robotics.Co.Ltd. All rights reserved.

For further information, contact: contact@bridgedp.com or visit our website
at www.bridgedp.com.
********************************************************************************/

#include <legged_hw/LeggedHWLoop.h>
#include "dora2HW.h"

int main(int argc, char** argv)
{
  // 初始化ROS节点
  ros::init(argc, argv, "legged_dora2_hw");
  // 创建ROS节点句柄
  ros::NodeHandle nh;
  // 创建用于访问私有命名空间参数的节点句柄
  ros::NodeHandle robot_hw_nh("~");

  // 创建并启动一个异步旋转器，允许多线程处理回调
  ros::AsyncSpinner spinner(3); // 使用3个线程
  spinner.start();

  try
  {
    // 创建硬件接口的共享指针实例
    std::shared_ptr<legged::dora2HW> legged_dora2_hw = std::make_shared<legged::dora2HW>();

    // 初始化硬件接口
    legged_dora2_hw->init(nh, robot_hw_nh);std::cout<<"legged_dora2_hw line"<<__LINE__<<std::endl;
    // 通过调用LeggedHWLoop类的构造函数，我们创建了一个名为control_loop的LeggedHWLoop对象，这个对象会启动硬件控制循环
    legged::LeggedHWLoop control_loop(nh, legged_dora2_hw);std::cout<<"legged_dora2_hw line"<<__LINE__<<std::endl;
    std::cout<<"legged_dora2_hw line"<<__LINE__<<std::endl;
    // 等待ROS节点关闭
    ros::waitForShutdown();
  }
  catch (const ros::Exception& e)
  {
    // 如果发生异常，打印错误信息并退出
    ROS_FATAL_STREAM("Error in the hardware interface:\n"
                     << "\t" << e.what());
    return 1; // 返回1表示异常退出
  }
  printf("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
  return 0; // 正常退出
}

