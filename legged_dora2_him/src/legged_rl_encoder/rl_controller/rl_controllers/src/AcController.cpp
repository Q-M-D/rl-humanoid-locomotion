#include "rl_controllers/AcController.h"
#include <pluginlib/class_list_macros.hpp>
#include "rl_controllers/RotationTools.h"
#include <algorithm>

namespace legged
{

  void AcController::handleWalkMode()
  {
    // compute observation & actions
    if (std::cout.fail())
    {
      std::cerr << "std::cout is in a bad state!" << std::endl;
      // 可能需要清除错误状态
      std::cout.clear();
    }
    if (loopCount_ % robotCfg_.controlCfg.decimation == 0)
    {
      computeObservation();
      computeActions();
      // limit action range
      scalar_t actionMin = -robotCfg_.clipActions;
      scalar_t actionMax = robotCfg_.clipActions;
      std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                     [actionMin, actionMax](scalar_t x)
                     { return std::max(actionMin, std::min(actionMax, x)); });
    }

    // set action
    for (int i = 0; i < actionsSize_; i++)
    {
      std::string partName = hybridJointHandles_[i].getName();
      scalar_t pos_des = actions_[i] * robotCfg_.controlCfg.actionScale + defaultJointAngles_(i);
      double stiffness = robotCfg_.controlCfg.stiffness[partName]; // 根据关节名称获取刚度
      double damping = robotCfg_.controlCfg.damping[partName]; // 根据关节名称获取阻尼
      std::cout << "joint_name:" << partName << "kp:" << stiffness << " kd:" << damping << std::endl;
      hybridJointHandles_[i].setCommand(pos_des, 0, stiffness, damping, 0);
      std::cout << "action:" << actions_[i] << std::endl;
      lastActions_(i, 0) = actions_[i];
    }
  }

  bool AcController::loadModel(ros::NodeHandle &nh)
  {
    std::string policyFilePath;
    if (!nh.getParam("/policyFile", policyFilePath))
    {
      ROS_ERROR_STREAM("Get policy path fail from param server, some error occur!");
      return false;
    }
    policyFilePath_ = policyFilePath;
    ROS_INFO_STREAM("Load Onnx model from path : " << policyFilePath);

    // create env
    onnxEnvPrt_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController"));
    // create session
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetInterOpNumThreads(1);
    policySessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPrt_, policyFilePath.c_str(), sessionOptions);
    // get input and output info
    policyInputNames_.clear();
    policyOutputNames_.clear();
    policyInputShapes_.clear();
    policyOutputShapes_.clear();

    Ort::AllocatorWithDefaultOptions allocator;
    ROS_INFO_STREAM("count: " << policySessionPtr_->GetOutputCount());
    for (int i = 0; i < policySessionPtr_->GetInputCount(); i++)
    {
      auto policyInputnamePtr = policySessionPtr_->GetInputNameAllocated(i, allocator);
      policyInputNodeNameAllocatedStrings.push_back(std::move(policyInputnamePtr));
      policyInputNames_.push_back(policyInputNodeNameAllocatedStrings.back().get());
      // inputNames_.push_back(sessionPtr_->GetInputNameAllocated(i, allocator).get());
      policyInputShapes_.push_back(policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
      std::vector<int64_t> policyShape = policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      std::cerr << "Policy Shape: [";
      for (size_t j = 0; j < policyShape.size(); ++j)
      {
          std::cout << policyShape[j];
          if (j != policyShape.size() - 1)
          {
              std::cerr << ", ";
          }
      }
      std::cout << "]" << std::endl;
}
    for (int i = 0; i < policySessionPtr_->GetOutputCount(); i++)
    {
      auto policyOutputnamePtr = policySessionPtr_->GetOutputNameAllocated(i, allocator);
      policyOutputNodeNameAllocatedStrings.push_back(std::move(policyOutputnamePtr));
      policyOutputNames_.push_back(policyOutputNodeNameAllocatedStrings.back().get());
      // outputNames_.push_back(sessionPtr_->GetOutputNameAllocated(i, allocator).get());
      std::cout << policySessionPtr_->GetOutputNameAllocated(i, allocator).get() << std::endl;
      policyOutputShapes_.push_back(policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
      std::vector<int64_t> policyShape = policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      std::cerr << "Policy Shape: [";
      for (size_t j = 0; j < policyShape.size(); ++j)
      {
          std::cout << policyShape[j];
          if (j != policyShape.size() - 1)
          {
              std::cerr << ", ";
          }
      }
      std::cout << "]" << std::endl;
    }

    ROS_INFO_STREAM("Load Onnx model successfully !!!");
    return true;
  }

  bool AcController::loadRLCfg(ros::NodeHandle &nh)
  {
    RLRobotCfg::InitState &initState = robotCfg_.initState;
    RLRobotCfg::ControlCfg &controlCfg = robotCfg_.controlCfg;
    RLRobotCfg::ObsScales &obsScales = robotCfg_.obsScales;

    int error = 0;
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/leg_l1_joint", initState.leg_l1_joint));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/leg_l2_joint", initState.leg_l2_joint));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/leg_l3_joint", initState.leg_l3_joint));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/leg_l4_joint", initState.leg_l4_joint));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/leg_l5_joint", initState.leg_l5_joint));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/leg_l5_joint", initState.leg_l6_joint));


    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/leg_r1_joint", initState.leg_r1_joint));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/leg_r2_joint", initState.leg_r2_joint));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/leg_r3_joint", initState.leg_r3_joint));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/leg_r4_joint", initState.leg_r4_joint));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/leg_r5_joint", initState.leg_r5_joint));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/init_state/default_joint_angle/leg_r5_joint", initState.leg_r6_joint));
    

    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/stiffness", controlCfg.stiffness));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/damping", controlCfg.damping));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/action_scale", controlCfg.actionScale));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/decimation", controlCfg.decimation));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/control/cycle_time", controlCfg.cycle_time));

    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/clip_scales/clip_observations", robotCfg_.clipObs));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/clip_scales/clip_actions", robotCfg_.clipActions));

    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/lin_vel", obsScales.linVel));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/ang_vel", obsScales.angVel));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/dof_pos", obsScales.dofPos));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/dof_vel", obsScales.dofVel));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/height_measurements", obsScales.heightMeasurements));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/normalization/obs_scales/quat", obsScales.quat));

    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/actions_size", actionsSize_));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/observations_size", observationSize_));
    error += static_cast<int>(!nh.getParam("/LeggedRobotCfg/size/stack_size", stackSize_));

    actions_.resize(actionsSize_);
    policyObservations_.resize(observationSize_ * stackSize_);
    std::fill(policyObservations_.begin(), policyObservations_.end(), 0.0f);

    command_.x = 0;
    command_.y = 0;
    command_.yaw = 0;
    baseLinVel_.setZero();
    basePosition_.setZero();
    std::vector<scalar_t> defaultJointAngles{
        robotCfg_.initState.leg_l1_joint, robotCfg_.initState.leg_l2_joint, robotCfg_.initState.leg_l3_joint,
        robotCfg_.initState.leg_l4_joint, robotCfg_.initState.leg_l5_joint, robotCfg_.initState.leg_l6_joint,
        robotCfg_.initState.leg_r1_joint,robotCfg_.initState.leg_r2_joint, robotCfg_.initState.leg_r3_joint,
        robotCfg_.initState.leg_r4_joint,robotCfg_.initState.leg_r5_joint, robotCfg_.initState.leg_r6_joint,};
    lastActions_.resize(actionsSize_);
    lastActions_.setZero();
    const int inputSize = stackSize_ * observationSize_;
    proprioHistoryBuffer_.resize(inputSize);
    defaultJointAngles_.resize(actuatedDofNum_);
    for (int i = 0; i < actuatedDofNum_; i++)
    {
      defaultJointAngles_(i) = defaultJointAngles[i];
    }

    return (error == 0);
  }

  void AcController::computeActions()
  {

    std::vector<Ort::Value> policyInputValues;
    policyInputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, policyObservations_.data(), policyObservations_.size(),
                                                                         policyInputShapes_[0].data(), policyInputShapes_[0].size()));
    // run inference
    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputValues = policySessionPtr_->Run(runOptions, policyInputNames_.data(), policyInputValues.data(), 1, policyOutputNames_.data(), 1);

    if (isfirstCompAct_){
      for (int i = 0; i < policyObservations_.size(); ++i) {
        std::cout << policyObservations_[i] << " ";
        if ((i + 1) % observationSize_ == 0) {
            std::cout << std::endl;
        }
      }
      isfirstCompAct_ = false;
    }

    for (int i = 0; i < actionsSize_; i++)
    {
      actions_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
    }
  }

  void AcController::computeObservation()
  {
    RLRobotCfg::ObsScales &obsScales = robotCfg_.obsScales;
    // command
    vector_t command(5);
    phase_ = phase_ / robotCfg_.controlCfg.cycle_time;

    // command_x = 0.995 * command_x + 0.005 * command_.x;

    // command_y = 0.995 * command_y + 0.005 * command_.y;

    // command_yaw = 0.995 * command_yaw + 0.005 * command_.yaw;

    if (command_x < -0.5) command_x = -0.5;
    // if (abs(command_x) < 0.05 && abs(command_y) < 0.05 && abs(command_.yaw) < 0.05){
    //   phase_ = 0;
    //   // command_.x = 0.0;
    //   // command_.y = 0.0;
    //   // command_.yaw = 0.0;
    // }


    command[0] = sin(2 * M_PI * phase_);
    command[1] = cos(2 * M_PI * phase_);
    command[2] = command_.x* obsScales.linVel;
    command[3] = command_.y * obsScales.linVel;
    command[4] = command_.yaw * obsScales.angVel;


    // actions
    vector_t actions(lastActions_);

    matrix_t commandScaler = Eigen::DiagonalMatrix<scalar_t, 3>(obsScales.linVel, obsScales.linVel, obsScales.angVel);

    vector_t proprioObs(observationSize_);

    proprioObs << command, // 5
        propri_.baseAngVel * obsScales.angVel,  // 3
        propri_.baseEulerXyz(0) * obsScales.quat,  // 1
        propri_.baseEulerXyz(1) * obsScales.quat,  // 1
        (propri_.jointPos - defaultJointAngles_) * obsScales.dofPos,  // 12
        propri_.jointVel * obsScales.dofVel,  // 12
        actions;  // 12


    if (isfirstRecObs_)
    {
      for (
         int i = 34; i <46; i++) //observationSize_: 46, actionSize_: 12
      {
        proprioObs(i,0) = 0.0;
      }

      for (size_t i = 0; i < stackSize_; i++)
      {
        proprioHistoryBuffer_.segment(i * observationSize_, observationSize_) = proprioObs.cast<tensor_element_t>();
      }
      isfirstRecObs_ = false;
    }

    proprioHistoryBuffer_.head(proprioHistoryBuffer_.size() - observationSize_) =
        proprioHistoryBuffer_.tail(proprioHistoryBuffer_.size() - observationSize_);
    proprioHistoryBuffer_.tail(observationSize_) = proprioObs.cast<tensor_element_t>();


    // clang-format on

    for (size_t i = 0; i < (observationSize_ * stackSize_); i++){
      policyObservations_[i] = static_cast<tensor_element_t>(proprioHistoryBuffer_[i]);
      // if(i < observationSize_)
      // std::cout << i << "obs:::" << estObservations_[i] << std::endl;
    }
    // Limit observation range
    scalar_t obsMin = -robotCfg_.clipObs;
    scalar_t obsMax = robotCfg_.clipObs;
    std::transform(policyObservations_.begin(), policyObservations_.end(), policyObservations_.begin(),
                   [obsMin, obsMax](scalar_t x)
                   { return std::max(obsMin, std::min(obsMax, x)); });
  }

} // namespace legged

PLUGINLIB_EXPORT_CLASS(legged::AcController, controller_interface::ControllerBase)