//
// Created by han on 23-2-15.
//

#include "FSM_State.h"
#include "iostream"

FSM_State::FSM_State(const std::shared_ptr<ControlFSMData>& controlFSM_data, FSM_StateName state_name_in,
                     std::string state_string_in):
                     data_(controlFSM_data),
                     state_name_(state_name_in),
                     state_string_(state_string_in)
{
    transition_data_.zero();
    std::cout << "[FSM_State] Initialized FSM state: " << state_string_in
              << std::endl;
}
