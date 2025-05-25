//
// Created by han on 23-2-22.
//

#include "FSM_State_Passive.h"

FSM_State_Passive::FSM_State_Passive(const std::shared_ptr<ControlFSMData>& control_data)
        : FSM_State(control_data, FSM_StateName::PASSIVE, "PASSIVE") {
    // Do nothing
    // only check encoder
}


void FSM_State_Passive::onEnter() {
    // Default is to not transition
    this->next_state_name_ = this->state_name_;

    // Reset the transition data
    this->transition_data_.zero();

    this->data_->state_no_ = K_PASSIVE;
}

/**
 * Calls the functions to be executed on each control loop iteration.
 */

void FSM_State_Passive::run() {
    // Do nothing, all commands should begin as zeros
    // testTransition();
}

void FSM_State_Passive::onExit() {
    // Nothing to clean up when exiting
}

TransitionData FSM_State_Passive::transition() {
    // Finish Transition
    this->transition_data_.done = true;

    // Return the transition data to the FSM
    return this->transition_data_;
}

FSM_StateName FSM_State_Passive::checkTransition() {
    this->next_state_name_ = this->state_name_;
    iter++;

    // get motor encoder raw data
    if(iter > 5000)
    {
        this->next_state_name_ = FSM_StateName::TEST;
    }

    // Get the next state
    return this->next_state_name_;
}
