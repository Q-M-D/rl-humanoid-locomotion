#pragma once

#include "rl_controllers/Types.h"
#include <robot_state_publisher/robot_state_publisher.h>

#include <controller_interface/multi_interface_controller.h>
#include <gazebo_msgs/ModelStates.h>
#include <hardware_interface/imu_sensor_interface.h>
#include <legged_common/hardware_interface/ContactSensorInterface.h>
#include <legged_common/hardware_interface/HybridJointInterface.h>

#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32.h>
#include <tf/transform_broadcaster.h>

#include <controller_manager_msgs/SwitchController.h>
#include <sensor_msgs/Joy.h>

#include <onnxruntime/onnxruntime_cxx_api.h>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "TutorialsConfig.h"
#include <dynamic_reconfigure/server.h>
#include <dynamic_reconfigure/ParamDescription.h>

#include <atomic>

namespace legged
{

  struct RLRobotCfg
  {
    struct ControlCfg
    {
      std::map<std::string, float> stiffness;
      std::map<std::string, float> damping;
      float actionScale;
      int decimation;
      float user_torque_limit;
      float user_power_limit;
      float cycle_time;
    };

    struct InitState
    {
      // default joint angles  listao
      scalar_t leg_l1_joint;
      scalar_t leg_l2_joint;
      scalar_t leg_l3_joint;
      scalar_t leg_l4_joint;
      scalar_t leg_l5_joint;
      scalar_t leg_l6_joint;
      scalar_t leg_r1_joint;
      scalar_t leg_r2_joint;
      scalar_t leg_r3_joint;
      scalar_t leg_r4_joint;
      scalar_t leg_r5_joint;
      scalar_t leg_r6_joint;
    };

    struct ObsScales
    {
      scalar_t linVel;
      scalar_t angVel;
      scalar_t dofPos;
      scalar_t dofVel;
      scalar_t quat;
      scalar_t heightMeasurements;
    };

    bool encoder_nomalize;

    scalar_t clipActions;
    scalar_t clipObs;

    InitState initState;
    ObsScales obsScales;
    ControlCfg controlCfg;
  };
  // listao
  struct JointState
  {
    scalar_t leg_l1_joint;
    scalar_t leg_l2_joint;
    scalar_t leg_l3_joint;
    scalar_t leg_l4_joint;
    scalar_t leg_l5_joint;
    scalar_t leg_l6_joint;
    scalar_t leg_r1_joint;
    scalar_t leg_r2_joint;
    scalar_t leg_r3_joint;
    scalar_t leg_r4_joint;
    scalar_t leg_r5_joint;
    scalar_t leg_r6_joint;
  };

  struct JoyInfo
  {
    float axes[8];
    int buttons[11];
  };

  struct Proprioception
  {
    vector_t jointPos;
    vector_t jointVel;
    vector3_t baseAngVel;
    vector3_t baseEulerXyz;
    // vector3_t baseAngZyx;  // base angular pos eular (zyx)
    // quaternion_t baseAngQuat;  // base angular pos quat (zyx)
    vector3_t projectedGravity;
  };

  struct Command
  {
    std::atomic<scalar_t> x;
    std::atomic<scalar_t> y;
    std::atomic<scalar_t> yaw;
  };

  class RLControllerBase : public controller_interface::MultiInterfaceController<HybridJointInterface, hardware_interface::ImuSensorInterface,
                                                                                 ContactSensorInterface>
  {
  public:
    enum class Mode : uint8_t
    {
      LIE,
      STAND,
      WALK,
      DEFAULT
    };

    RLControllerBase() = default;
    virtual ~RLControllerBase() = default;
    virtual bool init(hardware_interface::RobotHW *robotHw, ros::NodeHandle &controllerNH);
    virtual void starting(const ros::Time &time);
    virtual void update(const ros::Time &time, const ros::Duration &period);

    virtual bool loadModel(ros::NodeHandle &nh) { return false; };
    virtual bool loadRLCfg(ros::NodeHandle &nh) { return false; };
    virtual void computeActions(){};
    virtual void computeObservation(){};

    virtual void handleLieMode();
    virtual void handleStandMode();
    virtual void handleDefautMode();
    virtual void handleWalkMode(){};

    std::unique_ptr<dynamic_reconfigure::Server<legged_debugger::TutorialsConfig>> server_ptr_;
    void dynamicParamCallback(legged_debugger::TutorialsConfig &config, uint32_t level);

  protected:
    virtual void updateStateEstimation(const ros::Time &time, const ros::Duration &period);

    virtual void cmdVelCallback(const geometry_msgs::Twist &msg);
    virtual void joyInfoCallback(const sensor_msgs::Joy &msg);

    Mode mode_;
    int64_t loopCount_;
    Command command_;
    RLRobotCfg robotCfg_{};
    // listao
    JointState standjointState_{0.0, 0.0, -0.0,
                                0.0, -0.0, 0.0,
                                0.0, 0.0, -0.0,
                                0.0, -0.0, 0.0};

    JointState liejointState_{0.0, 0.0,-0.0,
                                0.0, -0.0, 0.0,
                                0.0, 0.0, -0.0,
                                0.0, -0.0, 0.0 };
    JoyInfo joyInfo;
    std::atomic_bool emergency_stop{false};
    std::atomic_bool start_control{false};
    std::atomic_bool position_control{false};
    ros::Time switchTime;

    vector_t rbdState_;
    vector_t measuredRbdState_;
    Proprioception propri_;

    // hardware interface
    std::vector<HybridJointHandle> hybridJointHandles_;
    hardware_interface::ImuSensorHandle imuSensorHandles_;
    std::vector<ContactSensorHandle> contactHandles_;

    ros::Subscriber cmdVelSub_;
    ros::Subscriber joyInfoSub_;
    ros::Subscriber emgStopSub_;
    ros::Subscriber startCtrlSub_;
    ros::Subscriber switchModeSub_;
    ros::Subscriber walkModeSub_;
    ros::Subscriber positionCtrlSub_;
    controller_manager_msgs::SwitchController switchCtrlSrv_;
    ros::ServiceClient switchCtrlClient_;
    // listao
    int actuatedDofNum_ = 12;

    ros::Publisher realJointVelPublisher_;
    ros::Publisher realJointPosPublisher_;
    ros::Publisher realTorquePublisher_;
    std::unique_ptr<robot_state_publisher::RobotStatePublisher> robotStatePublisherPtr_;

    ros::Publisher realImuAngularVelPublisher_;
    ros::Publisher realImuLinearAccPublisher_;
    ros::Publisher realImuEulerXyzPulbisher;

    ros::Publisher outputPlannedJointVelPublisher_;
    ros::Publisher outputPlannedJointPosPublisher_;
    ros::Publisher outputPlannedTorquePublisher_;

    int walkCount_ = 0;
    float cfg_kd;
    double phase_;

  private:
    // PD stand

    vector_t pos_des_output_{};
    vector_t vel_des_output_{};

    size_t joint_dim_{0};

    std::vector<scalar_t> currentJointAngles_;
    vector_t standJointAngles_;
    vector_t lieJointAngles_;

    scalar_t standPercent_;
    scalar_t standDuration_;
    tf::TransformBroadcaster tfBroadcaster_;

  };
} // namespace legged