#pragma once

#include "cppTypes.h"
#include "log.h"
#include "RemoteUserParameter.h"
#include "interpolate.h"
#include <mutex>

class Frame {
public:
    Frame(const std::shared_ptr<RobotModel>& robot_model,
            const std::shared_ptr<RemoteUserParameter>& user_param) {
        if (robot_model == nullptr || user_param == nullptr) {
            std::cout << "construct frame meet nullptr" << std::endl;
            return;
        }
        const int single_arm_freedom = robot_model->single_arm_freedom;
        const int single_leg_freedom = robot_model->single_leg_freedom;
        robot_state = std::make_shared<RobotState>(single_arm_freedom, single_leg_freedom);
        joint_cmd_ = std::make_shared<JointCommand>(single_arm_freedom, single_leg_freedom);
        walk_phase_trans_data = std::make_shared<WalkPhaseTransitionData>(single_arm_freedom, single_leg_freedom);
        robot_model_ = robot_model;
        user_param_ = user_param; 
    }
    ~Frame() { }

    std::shared_ptr<RobotState> mutableRobotState() {
        return robot_state;
    }

    void setGlobalTime(const double& t) {
        global_time = t;
    }
    double getGlobalTime() const {return global_time;}

    std::shared_ptr<WalkPhaseTransitionData>& mutableWalkPhaseTransitionData() {
        return walk_phase_trans_data;
    }

    std::shared_ptr<RobotModel>& mutableRobotModel() {
        return robot_model_;
    }
    std::shared_ptr<RemoteUserParameter>& mutableUserParam() {
        return user_param_;
    }

    std::shared_ptr<Command>& mutableCommand() {
        return command_;
    }
    std::shared_ptr<JointCommand>& mutableJointCommand() {
        return joint_cmd_;
    }
    
private:
    std::shared_ptr<RobotState> robot_state = nullptr;
    std::shared_ptr<RobotModel> robot_model_ = nullptr;
    std::shared_ptr<RemoteUserParameter> user_param_ = nullptr;

    double global_time = 0.0;
    std::shared_ptr<WalkPhaseTransitionData> walk_phase_trans_data = nullptr;

    std::shared_ptr<Command> command_ = std::make_shared<Command>();
    std::shared_ptr<JointCommand> joint_cmd_ = nullptr;
};