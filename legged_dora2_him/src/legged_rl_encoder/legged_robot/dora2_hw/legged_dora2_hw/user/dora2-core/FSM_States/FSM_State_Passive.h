//
// Created by han on 23-2-22.
//

#ifndef BAER_ETHERCAT_FSM_STATE_PASSIVE_H
#define BAER_ETHERCAT_FSM_STATE_PASSIVE_H

#include "FSM_State.h"

class FSM_State_Passive: public FSM_State {
public:
    explicit FSM_State_Passive(const std::shared_ptr<ControlFSMData>& control_data);

    // Behavior to be carried out when entering a state
    void onEnter();

    // Run the normal behavior for the state
    void run();

    // Checks for any transition triggers
    FSM_StateName checkTransition();

    // Manages state specific transitions
    TransitionData transition();

    // Behavior to be carried out when exiting a state
    void onExit();

    //TransitionData testTransition();

private:
    // Keep track of the control iterations
    int iter = 0;
};


#endif //BAER_ETHERCAT_FSM_STATE_PASSIVE_H
