//
// Created by han on 23-2-16.
//

#ifndef BAER_ETHERCAT_CONTROLFSM_H
#define BAER_ETHERCAT_CONTROLFSM_H

#include "FSM_State.h"
#include "FSM_State_Passive.h"
#include "FSM_State_Test.h"

enum class FSM_OperatingMode {
    NORMAL, TRANSITIONING, ESTOP, EDAMP };

struct FSM_StatesList {
    std::shared_ptr<FSM_State> invalid;
    std::shared_ptr<FSM_State_Passive> passive;
    std::shared_ptr<FSM_State_Test> test;
};

class ControlFSM {
public:
    ControlFSM(const std::shared_ptr<ControlFSMData>& data);

    // Initializes the Control FSM instance
    void initialize();

    // Runs the FSM logic and handles the state transitions and normal runs
    void runFSM();

    // This will be removed and put into the SafetyChecker class
    FSM_OperatingMode safetyPreCheck();

    //
    FSM_OperatingMode safetyPostCheck();

    // Gets the next FSM_State from the list of created states when requested
    std::shared_ptr<FSM_State> getNextState(FSM_StateName stateName);

    // Prints the current FSM status
    void printInfo(int opt);

    // Contains all of the control related data
    std::shared_ptr<ControlFSMData> data_;


    // FSM state information
    FSM_StatesList states_list_;  // holds all of the FSM States
    std::shared_ptr<FSM_State> current_state_;    // current FSM state
    std::shared_ptr<FSM_State> next_state_;       // next FSM state
    FSM_StateName next_state_name_;   // next FSM state name

    // Checks all of the inputs and commands for safety
    // SafetyChecker safetyChecker;

    TransitionData transition_data_;

private:
    // Operating mode of the FSM
    FSM_OperatingMode operating_mode_;

    // Choose how often to print info, every N iterations
    int print_num_ = 10000;  // N*(0.001s) in simulation time

    // Track the number of iterations since last info print
    int print_iter_ = 0;  // make larger than printNum to not print

    int iter_ = 0;

    bool is_first_ = true;
};

#endif //BAER_ETHERCAT_CONTROLFSM_H
