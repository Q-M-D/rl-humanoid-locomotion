#pragma once 

#include "FSM_State.h"

class FSM_State_Test : public FSM_State {
public:
    FSM_State_Test(const std::shared_ptr<ControlFSMData>& fsm_data);

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

private:
    int iter_ = 0;
    Interpolate interpolate_;
};
