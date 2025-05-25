//
// Created by han on 23-2-16.
//

#include "ControlFSM.h"


ControlFSM::ControlFSM(const std::shared_ptr<ControlFSMData>& data){
    data_ = data;

    states_list_.invalid = nullptr;
    states_list_.passive = std::make_shared<FSM_State_Passive>(data);
    states_list_.test = std::make_shared<FSM_State_Test>(data);


    initialize();
}

/**
 * Initialize the Control FSM with the default settings. SHould be set to
 * Passive state and Normal operation mode.
 */

void ControlFSM::initialize() {
    current_state_ = states_list_.passive;

    current_state_->onEnter();

    next_state_ = current_state_;

    operating_mode_ = FSM_OperatingMode::NORMAL;

}

void ControlFSM::runFSM() {
    // 运行安全性预检查，并设置操作模式
    operating_mode_ = safetyPreCheck();

    // 如果是第一次运行，则记录当前时间作为起始时间
    if(is_first_){
        auto const now = std::chrono::high_resolution_clock::now();
        auto const second =
                std::chrono::duration_cast<std::chrono::microseconds>(
                        now.time_since_epoch()
                );

        data_->begin_time_ = static_cast<double>(second.count()) / 1000000.0;
        is_first_ = false;
    }

    // 记录当前时间，并计算运行时间
    auto const now = std::chrono::high_resolution_clock::now();
    auto const second =
            std::chrono::duration_cast<std::chrono::microseconds>(
                    now.time_since_epoch()
            );

    double current_time = static_cast<double>(second.count()) / 1000000.0;
    data_->time_ = current_time - data_->begin_time_;

    // 根据操作模式进行不同的处理
    if (operating_mode_ != FSM_OperatingMode::ESTOP) {
        if (operating_mode_ == FSM_OperatingMode::NORMAL) {
            // 检查状态转移
            next_state_name_ = current_state_->checkTransition();

            // 如果有状态转移，则设置操作模式为转换中，并获取下一个状态
            if (next_state_name_ != current_state_->state_name_) {
                operating_mode_ = FSM_OperatingMode::TRANSITIONING;
                next_state_ = getNextState(next_state_name_);
            } else {
                // 否则继续执行当前状态
                current_state_->run();
            }
        }

        // 如果操作模式为转换中
        if (operating_mode_ == FSM_OperatingMode::TRANSITIONING) {
            // 执行状态转移
            transition_data_ = current_state_->transition();

            // 进行安全性后检查
            safetyPostCheck();

            // 如果转移完成
            if (transition_data_.done) {
                // 退出当前状态
                current_state_->onExit();

                // 切换到下一个状态
                current_state_ = next_state_;

                // 进入新状态
                current_state_->onEnter();

                // 设置操作模式为正常模式
                operating_mode_ = FSM_OperatingMode::NORMAL;
            }
        } else {
            // 进行安全性后检查
            safetyPostCheck();
        }
    } else { // 如果是急停模式
        // 切换到被动状态并进入该状态
        current_state_ = states_list_.passive;
        current_state_->onEnter();
        next_state_name_ = current_state_->state_name_;
    }
}


FSM_OperatingMode ControlFSM::safetyPreCheck() {
    // Default is to return the current operating mode
    return operating_mode_;
}

FSM_OperatingMode ControlFSM::safetyPostCheck()
{
    return operating_mode_;
}

std::shared_ptr<FSM_State> ControlFSM::getNextState(FSM_StateName stateName)
{
    switch (stateName) {
        case FSM_StateName::INVALID:
            return states_list_.invalid;

        case FSM_StateName::PASSIVE:
            return states_list_.passive;

        /*case FSM_StateName::JOINT_PD:
            return states_list_.jointPD;*/

        /*case FSM_StateName::IMPEDANCE_CONTROL:
            return statesList.impedanceControl;

        case FSM_StateName::STAND_UP:
            return statesList.standUp;*/
        case FSM_StateName::TEST:
            return states_list_.test;

        default:
            return states_list_.invalid;
    }
}

