//
// Created by han on 23-2-15.
//

#ifndef BAER_ETHERCAT_FSM_STATE_H
#define BAER_ETHERCAT_FSM_STATE_H

#include "ControlFSMData.h"
#include "TransitionData.h"

#include <iostream>
#include "log.h"

// Normal robot states
#define K_PASSIVE 0
#define K_ZERO 2
#define K_TEST 3

// Specific control states
#define K_JOINT_PD 51
#define K_IMPEDANCE_CONTROL 52

#define K_INVALID 100

enum class FSM_StateName {
    INVALID,
    PASSIVE,
    TEST,
};

class FSM_State {
public:
    FSM_State(const std::shared_ptr<ControlFSMData>& controlFSM_data, FSM_StateName state_name_in,
              std::string state_string_in);

    // Behavior to be carried out when entering a state
    virtual void onEnter() = 0;// {}

    // Run the normal behavior for the state
    virtual void run() = 0; //{}

    // Manages state specific transitions
    virtual FSM_StateName checkTransition() { return FSM_StateName::INVALID; }

    // Runs the transition behaviors and returns true when done transitioning
    virtual TransitionData transition() { return transition_data_; }

    // Behavior to be carried out when exiting a state
    virtual void onExit() = 0; // {}


    // Holds all of the relevant control data
    std::shared_ptr<ControlFSMData> data_;

    // FSM State info
    FSM_StateName state_name_;      // enumerated name of the current state
    FSM_StateName next_state_name_;  // enumerated name of the next state
    std::string state_string_;      // state name string

    // Transition parameters
    double transition_duration_;  // transition duration time
    double t_start_transition_;    // time transition starts
    TransitionData transition_data_;

};


#endif //BAER_ETHERCAT_FSM_STATE_H
