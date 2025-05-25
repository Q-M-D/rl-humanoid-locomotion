#pragma once

#include "rl_controllers/RLControllerBase.h"

namespace legged
{

  class AcController : public RLControllerBase
  {
    using tensor_element_t = float;

  public:
    AcController() : memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)){}

    ~AcController() override = default;

  protected:
    bool loadModel(ros::NodeHandle &nh) override;
    bool loadRLCfg(ros::NodeHandle &nh) override;
    void computeActions() override;
    void computeObservation() override;
    void handleWalkMode() override;

  private:
    // onnx policy model
    std::string policyFilePath_;
    std::shared_ptr<Ort::Env> onnxEnvPrt_;
    std::unique_ptr<Ort::Session> policySessionPtr_;
    std::vector<const char *> policyInputNames_;
    std::vector<const char *> policyOutputNames_;
    std::vector<Ort::AllocatedStringPtr> policyInputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> policyOutputNodeNameAllocatedStrings;
    std::vector<std::vector<int64_t>> policyInputShapes_;
    std::vector<std::vector<int64_t>> policyOutputShapes_;

    std::string estFilePath_;
    std::unique_ptr<Ort::Session> estSessionPtr_;
    std::vector<const char *> estInputNames_;
    std::vector<const char *> estOutputNames_;
    std::vector<Ort::AllocatedStringPtr> estInputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> estOutputNodeNameAllocatedStrings;
    std::vector<std::vector<int64_t>> estInputShapes_;
    std::vector<std::vector<int64_t>> estOutputShapes_;

    vector3_t baseLinVel_;
    vector3_t basePosition_;
    vector_t lastActions_;
    vector_t defaultJointAngles_;

    bool isfirstRecObs_{true};
    int actionsSize_;
    int observationSize_;
    int stackSize_;
    std::vector<tensor_element_t> actions_;
    std::vector<tensor_element_t> latent_;
    std::vector<tensor_element_t> estObservations_;
    std::vector<tensor_element_t> policyObservations_;
    Ort::MemoryInfo memoryInfo;
    Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBuffer_;
    bool isfirstCompAct_{true};

    double command_x = 0.0;
    double command_y = 0.0;
    double command_yaw = 0.0;
  };

} // namespace legged