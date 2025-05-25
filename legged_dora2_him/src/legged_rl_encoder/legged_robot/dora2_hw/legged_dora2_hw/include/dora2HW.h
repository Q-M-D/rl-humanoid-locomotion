
//
// Created by hang on 5/10/24.
//

#pragma once

//legged_base
#include <legged_hw/LeggedHW.h>

// standard include
#include <iostream>
#include <fstream>
#include <ostream>
#include <memory.h>
#include <math.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <time.h>
#include <iomanip>
// msgs
#include <std_msgs/Int16MultiArray.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32.h>
#include "std_msgs/Float64MultiArray.h"
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <controller_manager_msgs/SwitchController.h>
// tf
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf2_ros/transform_listener.h>

// gaoqing_hw.h
#include <cstdio>
// #include "Console.hpp"
// #include "command.h"
// #include "transmit.h"
#include "Types.h"

// dora2 Host Computer Program include
#include "utilities.h"
#include "RemoteUserParameter.h"
#include "EthercatParameter.h"
#include "rt_ethercat_config.h"
#include "ControlFSMData.h"
#include "can_protocol_func.h"
#include "orientation_tools.h"

#include "Timer.h"
// #include "MotorConfig.h"
namespace legged
{

  struct EthercatOptions {
    std::string net_card;
    int ctr_freq;
    int robot_type;
    std::vector<int> node_motor_num;
    std::vector<int> motor_max_torque;
    std::vector<int> motor_min_torque;
    std::vector<int> motor_max_current;
    std::vector<int> motor_min_current;
    std::vector<int> right_arm_motor_type;
    std::vector<int> left_arm_motor_type;
    std::vector<int> right_leg_motor_type;
    std::vector<int> left_leg_motor_type;
    std::vector<int> right_leg_motor_dir;
    std::vector<int> left_leg_motor_dir;
    std::vector<int> right_arm_motor_dir;
    std::vector<int> left_arm_motor_dir;
  };

  

  const std::vector<std::string> CONTACT_SENSOR_NAMES = {"RF_FOOT", "LF_FOOT", "RH_FOOT", "LH_FOOT"};

  struct dora2MotorData
  {
    double pos_, vel_, tau_;                  // state
    double pos_des_, vel_des_, kp_, kd_, ff_; // command
  };

  struct dora2ImuData
  {
    double ori[4];
    double ori_cov[9];
    double angular_vel[3];
    double angular_vel_cov[9];
    double linear_acc[3];
    double linear_acc_cov[9];
  };

  class dora2HW : public LeggedHW
  {
  public:
    dora2HW()
    {
    }
    ~dora2HW()
    {
      std::cout << "~dora2HW_END" << std::endl;
    }

    /** \brief Get necessary params from param server. Init hardware_interface.
     *
     * Get params from param server and check whether these params are set. Load urdf of robot. Set up transmission and
     * joint limit. Get configuration of can bus and create data pointer which point to data received from Can bus.
     *
     * @param root_nh Root node-handle of a ROS node.
     * @param robot_hw_nh Node-handle for robot hardware.
     * @return True when init successful, False when failed.
     */
    bool init(ros::NodeHandle &root_nh, ros::NodeHandle &robot_hw_nh) override;

    /** \brief Communicate with hardware. Get data, status of robot.
     *
     * Call @ref Gsmp_LEGGED_SDK::UDP::Recv() to get robot's state.
     *
     * @param time Current time
     * @param period Current time - last time
     */
    void read(const ros::Time &time, const ros::Duration &period) override;

    /** \brief Comunicate with hardware. Publish command to robot.
     *
     * Propagate joint state to actuator state for the stored
     * transmission. Limit cmd_effort into suitable value. Call @ref Gsmp_LEGGED_SDK::UDP::Recv(). Publish actuator
     * current state.
     *
     * @param time Current time
     * @param period Current time - last time
     */
    void write(const ros::Time &time, const ros::Duration &period) override;

  private:
    void motor_setting(const std::shared_ptr<ControlFSMData>& a); 

    bool setupJoints();

    bool setupImu();

    bool setupContactSensor(ros::NodeHandle &nh);
      
    void CalibrateImu(const bool is_sim);

    std::shared_ptr<RobotModel> robot_model;
    std::shared_ptr<Frame> frame_ptr;
    std::shared_ptr<ControlFSMData> control_fsm_data;
    
    EthercatOptions config;
    dora2MotorData joint_data_[12]{};
    dora2ImuData imu_data_{};
    bool contact_state_[4]{};
    int contact_threshold_{};
    
    uint64_t hs = 0;
    Byte8 byte_8;
    Medulla* node_1;
    Medulla* node_2;
    Medulla* node_3;
    //Medulla* node_4;
    ImuRc* imu_rc;
    vector_t motor_pos_feedback_{12};
    vector_t motor_vel_feedback_{12};
    vector_t motor_tau_feedback_{12};
    vector_t joint_planned_torque_{12};
    ros::Subscriber odom_sub_;
    ros::Publisher motorPosPublisher_;
    ros::Publisher motorVelPublisher_;
    ros::Publisher motorTorquePublisher_;
    OrientationTools ori_tools_;

    tf::TransformListener *listener_;

    // void OdomCallBack(const sensor_msgs::Imu::ConstPtr &odom)
    // {
    //   yesenceIMU_ = *odom;
    // }

    // const vector<int> direction_motor{-1, -1, -1,
    //                                    1, -1, -1,
    //                                   -1,  1,  1,
    //                                    1,  1,  1};
    const std::vector<int> direction_motor{1, -1, 1, -1, 1, -1,
                                            1, -1, -1, 1, -1, -1};

    float bias_motor[12] = { 0, 0, 0.52, -1.05, 0.52, 0, 
                             0, 0, 0.52, -1.05, 0.52, 0};
    // float bias_motor[12] = { 0, 0, 0.123599, -0.39, 0.323599, 0, 
    //                          0, 0, 0.123599, -0.39, 0.323599, 0};
    // float bias_motor[12] = { 0, 0, 0., 0., 0., 0, 
    //                          0, 0., 0., 0.,0, 0};


  };

} // namespace legged
