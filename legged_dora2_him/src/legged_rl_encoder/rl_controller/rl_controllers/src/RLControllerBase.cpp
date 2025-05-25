#include "rl_controllers/RLControllerBase.h"
#include <string.h>
#include <pluginlib/class_list_macros.hpp>
#include "rl_controllers/RotationTools.h"
#include "rl_controllers/utilities.h"
#include <kdl_parser/kdl_parser.hpp>

namespace legged
{

  bool RLControllerBase::init(hardware_interface::RobotHW *robotHw, ros::NodeHandle &controllerNH)
  {
    // Hardware interface
    std::vector<std::string> jointNames{"leg_l1_joint", "leg_l2_joint", "leg_l3_joint","leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
                                        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint","leg_r4_joint", "leg_r5_joint", "leg_r6_joint",};
    std::vector<std::string> footNames{"leg_l5_link","leg_r5_link"};
    actuatedDofNum_ = jointNames.size();

    // Load policy model and rl cfg
    if (!loadModel(controllerNH))
    {
      ROS_ERROR_STREAM("[RLControllerBase] Failed to load the model. Ensure the path is correct and accessible.");
      return false;
    }
    if (!loadRLCfg(controllerNH))
    {
      ROS_ERROR_STREAM("[RLControllerBase] Failed to load the rl config. Ensure the yaml is correct and accessible.");
      return false;
    }
    standJointAngles_.resize(actuatedDofNum_);
    lieJointAngles_.resize(actuatedDofNum_);
    auto &StandState = standjointState_;
    auto &LieState = liejointState_;

    joint_dim_ = actuatedDofNum_;
    ros::NodeHandle nh;
    //加载kd
    nh.getParam("/Kd_config/kd", cfg_kd);

    urdf::Model urdfModel;
    if (!urdfModel.initParam("legged_robot_description")) {
      std::cerr << "[LeggedRobotVisualizer] Could not read URDF from: \"legged_robot_description\"" << std::endl;
    } else {
      KDL::Tree kdlTree;
      kdl_parser::treeFromUrdfModel(urdfModel, kdlTree);
      robotStatePublisherPtr_.reset(new robot_state_publisher::RobotStatePublisher(kdlTree));
    }
    // Topic init
    realJointPosPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/real_joint_pos", 1);
    realJointVelPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/real_joint_vel", 1);
    realTorquePublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/real_torque", 1);

    realImuAngularVelPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/imu_angular_vel", 1);
    realImuLinearAccPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/imu_linear_acc", 1);
    realImuEulerXyzPulbisher = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/imu_euler_xyz", 1);

    outputPlannedJointPosPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/rl_planned_joint_pos", 1);
    outputPlannedJointVelPublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/rl_planned_joint_vel", 1);
    outputPlannedTorquePublisher_ = nh.advertise<std_msgs::Float64MultiArray>("data_analysis/rl_planned_torque", 1);

    sensorContact_ = nh.advertise<std_msgs::Bool>("/contact_sensor_topic", 1);

    // Stand Lie joint // listao
    lieJointAngles_ <<  LieState.leg_l1_joint, LieState.leg_l2_joint, LieState.leg_l3_joint,LieState.leg_l4_joint, LieState.leg_l5_joint, LieState.leg_l6_joint,
                        LieState.leg_r1_joint, LieState.leg_r2_joint, LieState.leg_r3_joint,LieState.leg_r4_joint, LieState.leg_r5_joint, LieState.leg_r6_joint;

    // listao
    standJointAngles_ <<  StandState.leg_l1_joint, StandState.leg_l2_joint, StandState.leg_l3_joint,StandState.leg_l4_joint, StandState.leg_l5_joint, StandState.leg_l6_joint,
                        StandState.leg_r1_joint, StandState.leg_r2_joint, StandState.leg_r3_joint,StandState.leg_r4_joint, StandState.leg_r5_joint, StandState.leg_r6_joint;

    auto *hybridJointInterface = robotHw->get<HybridJointInterface>();
    for (const auto &jointName : jointNames)
    {
      hybridJointHandles_.push_back(hybridJointInterface->getHandle(jointName));
    }


    imuSensorHandles_ = robotHw->get<hardware_interface::ImuSensorInterface>()->getHandle("base_imu");

    cmdVelSub_ = controllerNH.subscribe("/cmd_vel", 1, &RLControllerBase::cmdVelCallback, this);
    joyInfoSub_ = controllerNH.subscribe("/joy", 1000, &RLControllerBase::joyInfoCallback, this);
    switchCtrlClient_ = controllerNH.serviceClient<controller_manager_msgs::SwitchController>("/controller_manager/switch_controller");
    auto emergencyStopCallback = [this](const std_msgs::Float32::ConstPtr &msg){emergency_stop = true;ROS_INFO("Emergency Stop");};
    emgStopSub_ = controllerNH.subscribe<std_msgs::Float32>("/emergency_stop", 1, emergencyStopCallback);

    // start control
    auto startControlCallback = [this](const std_msgs::Float32::ConstPtr &msg)
    {
      ros::Duration t(0.5);
      if (ros::Time::now() - switchTime > t)
      {
        if (!start_control)
        {
          start_control = true;
          standPercent_ = 0;
          for (size_t i = 0; i < hybridJointHandles_.size(); i++)
          {
            currentJointAngles_[i] = hybridJointHandles_[i].getPosition();
          }
          mode_ = Mode::LIE;
          ROS_INFO("Start Control");
        }
        else
        {
          start_control = false;
          mode_ = Mode::DEFAULT;
          ROS_INFO("ShutDown Control");
        }
        switchTime = ros::Time::now();
      }
    };
    startCtrlSub_ = controllerNH.subscribe<std_msgs::Float32>("/start_control", 1, startControlCallback);

    // switchMode
    auto switchModeCallback = [this](const std_msgs::Float32::ConstPtr &msg)
    {
      ros::Duration t(0.8);
      if (ros::Time::now() - switchTime > t)
      {
        if (start_control == true)
        {
          if (mode_ == Mode::STAND)
          {
            standPercent_ = 0;
            for (size_t i = 0; i < hybridJointHandles_.size(); i++)
            {
              currentJointAngles_[i] = hybridJointHandles_[i].getPosition();
            }
            mode_ = Mode::LIE;
            ROS_INFO("STAND2LIE");
          }
          else if (mode_ == Mode::LIE)
          {
            standPercent_ = 0;
            mode_ = Mode::STAND;
            ROS_INFO("LIE2STAND");
          }
        }
        switchTime = ros::Time::now();
      }
    };
    switchModeSub_ = controllerNH.subscribe<std_msgs::Float32>("/switch_mode", 1, switchModeCallback);

    // walkMode
    auto walkModeCallback = [this](const std_msgs::Float32::ConstPtr &msg)
    {
      ros::Duration t(0.2);
      if (ros::Time::now() - switchTime > t)
      {
        if (mode_ == Mode::STAND)
        {
          mode_ = Mode::WALK;
          ROS_INFO("STAND2WALK");
        }
        switchTime = ros::Time::now();
      }
    };
    walkModeSub_ = controllerNH.subscribe<std_msgs::Float32>("/walk_mode", 1, walkModeCallback);

    // positionMode
    auto positionModeCallback = [this](const std_msgs::Float32::ConstPtr &msg)
    {
      ros::Duration t(0.2);
      if (ros::Time::now() - switchTime > t)
      {
        if (mode_ == Mode::WALK)
        {
          mode_ = Mode::STAND;
          ROS_INFO("WALK2STAND");
        }
        else if (mode_ == Mode::DEFAULT)
        {
          standPercent_ = 0;
          for (size_t i = 0; i < hybridJointHandles_.size(); i++)
          {
            currentJointAngles_[i] = hybridJointHandles_[i].getPosition();
          }
          mode_ = Mode::LIE;
          ROS_INFO("DEF2LIE");
        }

        switchTime = ros::Time::now();
      }
    };
    positionCtrlSub_ = controllerNH.subscribe<std_msgs::Float32>("/position_control", 1, positionModeCallback);
    return true;
  }

  std::atomic<scalar_t> kp_stance{0};
  std::atomic<scalar_t> kd_stance{3};

  // only once
  void RLControllerBase::starting(const ros::Time &time)
  {
    updateStateEstimation(time, ros::Duration(0.002));
    currentJointAngles_.resize(hybridJointHandles_.size());
    scalar_t durationSecs = 2.0;
    standDuration_ = durationSecs * 500.0;
    standPercent_ = 0;
    mode_ = Mode::DEFAULT;
    loopCount_ = 0;

    server_ptr_ = std::make_unique<dynamic_reconfigure::Server<legged_debugger::TutorialsConfig>>(ros::NodeHandle("controller"));
    dynamic_reconfigure::Server<legged_debugger::TutorialsConfig>::CallbackType f;
    f = boost::bind(&RLControllerBase::dynamicParamCallback, this, _1, _2);
    server_ptr_->setCallback(f);

    pos_des_output_.resize(joint_dim_);
    vel_des_output_.resize(joint_dim_);
    pos_des_output_.setZero();
    vel_des_output_.setZero();

  }

  void RLControllerBase::update(const ros::Time &time, const ros::Duration &period)
  {
    ros::NodeHandle nh;
    updateStateEstimation(time, period);
    // ROS_WARN(mode: %d, mode_);
    switch (mode_)
    {
    case Mode::DEFAULT:
      handleDefautMode();
      break;
    case Mode::LIE:
      handleLieMode();
      break;
    case Mode::STAND:
      handleStandMode();
      break;
    case Mode::WALK:
      handleWalkMode();
      // if(walkCount_ > 3)
      //   exit(0);
      break;
    default:
      ROS_ERROR_STREAM("Unexpected mode encountered: " << static_cast<int>(mode_));
      break;
    }
    if (emergency_stop)
    {
      emergency_stop = false;
      mode_ = Mode::DEFAULT;
    }
    // if (emergency_stop && start_control)
    // {
    //   emergency_stop = false;
    //   starting(time);
    // }

    vector_t output_torque(joint_dim_);
    for (int j = 0; j < hybridJointHandles_.size(); j++)
    {
      pos_des_output_(j) = hybridJointHandles_[j].getPositionDesired();
      vel_des_output_(j) = hybridJointHandles_[j].getVelocityDesired();
      output_torque(j) = hybridJointHandles_[j].getFeedforward() +
                          hybridJointHandles_[j].getKp() * (hybridJointHandles_[j].getPositionDesired() - hybridJointHandles_[j].getPosition()) +
                          hybridJointHandles_[j].getKd() * (hybridJointHandles_[j].getVelocityDesired() - hybridJointHandles_[j].getVelocity());
    }
    outputPlannedJointPosPublisher_.publish(createFloat64MultiArrayFromVector(pos_des_output_));
    outputPlannedJointVelPublisher_.publish(createFloat64MultiArrayFromVector(vel_des_output_));
    outputPlannedTorquePublisher_.publish(createFloat64MultiArrayFromVector(output_torque));
    bool contact_state = false;
    legged::ContactSensorHandle contact_sensor("foot_contact_sensor", &contact_state);
    bool is_contact = contact_sensor.isContact();
    std_msgs::Bool msg;;
    sensorContact_.publish(msg);
    loopCount_++;
  }

  void RLControllerBase::handleDefautMode()
  {
    for (int j = 0; j < hybridJointHandles_.size(); j++)
      hybridJointHandles_[j].setCommand(0, 0, 0, 0.1, 0);

    // ROS_WARN(The value of kdConfig.cfg_kd is: %f, kdConfig.cfg_kd);
  }

  void RLControllerBase::handleLieMode()
  {
    if (standPercent_ <= 1)
    {
      for (int j = 0; j < hybridJointHandles_.size(); j++)
      {
        
        
        if(j == 4 || j == 5 || j == 10 || j == 11){
          scalar_t pos_des = currentJointAngles_[j] * (1 - standPercent_) + lieJointAngles_[j] * standPercent_;

          hybridJointHandles_[j].setCommand(pos_des, 0, 50, 0.01, 0);

        }
        else{
          scalar_t pos_des = currentJointAngles_[j] * (1 - standPercent_) + lieJointAngles_[j] * standPercent_;

          hybridJointHandles_[j].setCommand(pos_des, 0, 100, 2, 0);
        }
      }
      standPercent_ += 1 / standDuration_;
      standPercent_ = std::min(standPercent_, scalar_t(1));
    }
  }

  void RLControllerBase::handleStandMode()
  {
    if (standPercent_ <= 1)
    {
      for (int j = 0; j < hybridJointHandles_.size(); j++)
      {
        if(j == 4 || j == 5 || j == 10 || j == 11){
          scalar_t pos_des = lieJointAngles_[j] * (1 - standPercent_) + standJointAngles_[j] * standPercent_;
          

          hybridJointHandles_[j].setCommand(pos_des, 0, 50, 0.01, 0);
        }
        else{
          scalar_t pos_des = lieJointAngles_[j] * (1 - standPercent_) + standJointAngles_[j] * standPercent_;
          hybridJointHandles_[j].setCommand(pos_des, 0, 60, 2, 0);
        }
      }
      standPercent_ += 1 / standDuration_;
      standPercent_ = std::min(standPercent_, scalar_t(1));
    }
  }

  void RLControllerBase::updateStateEstimation(const ros::Time &time, const ros::Duration &period)
  {
    vector_t jointPos(hybridJointHandles_.size()), jointVel(hybridJointHandles_.size()), jointTor(hybridJointHandles_.size());
    vector_t imuEulerXyz(3);
    contact_flag_t contacts;
    quaternion_t quat;
    vector3_t angularVel, linearAccel;
    matrix3_t orientationCovariance, angularVelCovariance, linearAccelCovariance;
    for (size_t i = 0; i < hybridJointHandles_.size(); ++i)
    {
      // printf("%s\n", hybridJointHandles_[i].getName().c_str());
      jointPos(i) = hybridJointHandles_[i].getPosition();
      jointVel(i) = hybridJointHandles_[i].getVelocity();
      jointTor(i) = hybridJointHandles_[i].getEffort();
    }
    for (size_t i = 0; i < 4; ++i)
    {
      quat.coeffs()(i) = imuSensorHandles_.getOrientation()[i];
    }
    for (size_t i = 0; i < 3; ++i)
    {
      angularVel(i) = imuSensorHandles_.getAngularVelocity()[i];
      linearAccel(i) = imuSensorHandles_.getLinearAcceleration()[i];
    }
    for (size_t i = 0; i < 9; ++i)
    {
      orientationCovariance(i) = imuSensorHandles_.getOrientationCovariance()[i];
      angularVelCovariance(i) = imuSensorHandles_.getAngularVelocityCovariance()[i];
      linearAccelCovariance(i) = imuSensorHandles_.getLinearAccelerationCovariance()[i];
    }

    propri_.jointPos = jointPos;
    propri_.jointVel = jointVel;
    propri_.baseAngVel = angularVel;

    vector3_t gravityVector(0, 0, -1);
    vector3_t zyx = quatToZyx(quat);
    matrix_t inverseRot = getRotationMatrixFromZyxEulerAngles(zyx).inverse();
    propri_.projectedGravity = inverseRot * gravityVector;
    propri_.baseEulerXyz = quatToXyz(quat);
    phase_ = time.toSec();

    for (size_t i = 0; i < 3; ++i)
    {
      imuEulerXyz(i) = propri_.baseEulerXyz[i];
    }
    //Publish The data
    realImuAngularVelPublisher_.publish(createFloat64MultiArrayFromVector(angularVel));
    realImuLinearAccPublisher_.publish(createFloat64MultiArrayFromVector(linearAccel));
    realImuEulerXyzPulbisher.publish(createFloat64MultiArrayFromVector(imuEulerXyz));

    realTorquePublisher_.publish(createFloat64MultiArrayFromVector(jointTor));
    realJointPosPublisher_.publish(createFloat64MultiArrayFromVector(jointPos));
    realJointVelPublisher_.publish(createFloat64MultiArrayFromVector(jointVel));

    robotStatePublisherPtr_->publishFixedTransforms(true);
    tf::Transform baseTransform;
    baseTransform.setOrigin(tf::Vector3(0.0, 0.0, 0.0)); // Origin
    baseTransform.setRotation(tf::Quaternion(quat.coeffs()(0), quat.coeffs()(1), quat.coeffs()(2), quat.coeffs()(3)));  
    tfBroadcaster_.sendTransform(tf::StampedTransform(baseTransform, time, "world", "base_link"));

  std::map<std::string, scalar_t> jointPositions{
      {"leg_l1_joint", jointPos(0)}, {"leg_l2_joint", jointPos(1)}, {"leg_l3_joint", jointPos(2)}, 
      {"leg_l4_joint", jointPos(3)}, {"leg_l5_joint", jointPos(4)}, {"leg_l6_joint", jointPos(5)}, 
      {"leg_r1_joint", jointPos(6)}, {"leg_r2_joint", jointPos(7)}, {"leg_r3_joint", jointPos(8)}, 
      {"leg_r4_joint", jointPos(9)}, {"leg_r5_joint", jointPos(10)}, {"leg_r6_joint", jointPos(11)}};
    // printf(jointPositions);
   robotStatePublisherPtr_->publishTransforms(jointPositions, time);
  }

  void RLControllerBase::cmdVelCallback(const geometry_msgs::Twist &msg)
  {
    command_.x = msg.linear.x;
    command_.y = msg.linear.y;
    command_.yaw = msg.angular.z;
  }

  void RLControllerBase::dynamicParamCallback(legged_debugger::TutorialsConfig &config, uint32_t level)
  {
    kp_stance = config.kp_stance;
    kd_stance = config.kd_stance;
  }

  void RLControllerBase::joyInfoCallback(const sensor_msgs::Joy &msg)
  {
    if (msg.header.frame_id.empty())
    {
      return;
    }
    for (int i = 0; i < msg.axes.size(); i++)
    {
      joyInfo.axes[i] = msg.axes[i];
    }
    for (int i = 0; i < msg.buttons.size(); i++)
    {
      joyInfo.buttons[i] = msg.buttons[i];
    }
  }
} // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::RLControllerBase, controller_interface::ControllerBase)