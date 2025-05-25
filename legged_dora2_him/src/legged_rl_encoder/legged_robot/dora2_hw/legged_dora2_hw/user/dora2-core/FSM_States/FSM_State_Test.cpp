#include "FSM_State_Test.h"

FSM_State_Test::FSM_State_Test(const std::shared_ptr<ControlFSMData>& fsm_data)
        : FSM_State(fsm_data, FSM_StateName::TEST, "TEST") {
}

void FSM_State_Test::onEnter() {
    this->next_state_name_ = this->state_name_;
    this->transition_data_.zero();

    const auto& frame_ptr = data_->frame_ptr_;
    const auto& robot_state = frame_ptr->mutableRobotState();
    robot_state->test_phase = TestPhase::WAITING;
    const auto& test_phase_trans_data = frame_ptr->mutableWalkPhaseTransitionData();
    test_phase_trans_data->test_phase = TestPhase::WAITING;
    test_phase_trans_data->last_phase_end_time = frame_ptr->getGlobalTime();
    test_phase_trans_data->last_phase_end_robot_state = *robot_state;

    std::cout << "Enter Test FSM\n";
}

void FSM_State_Test::run() {
    const auto& frame_ptr = data_->frame_ptr_;
    const auto& robot_state = frame_ptr->mutableRobotState();
    const auto& user_param = frame_ptr->mutableUserParam();
    auto& test_phase_trans_data = frame_ptr->mutableWalkPhaseTransitionData();
    auto& robot_cmd = frame_ptr->mutableCommand()->robot_cmd;

    const double time_from_last_phase = frame_ptr->getGlobalTime() - test_phase_trans_data->last_phase_end_time;

    switch(robot_state->test_phase) {
        case TestPhase::WAITING: {
            if (robot_cmd.is_rotate_to_nonzero_pos) {
                std::cout << "from waiting to rotate_to_nonzero_pos\n";
                robot_state->test_phase = TestPhase::ROATE_TO_NONZERO_POS;
                test_phase_trans_data->last_phase_end_time = frame_ptr->getGlobalTime();
                test_phase_trans_data->last_phase_end_robot_state = *robot_state;

                robot_cmd.is_rotate_to_nonzero_pos = false;
                robot_cmd.is_stop = false;
            } else if (robot_cmd.is_rotate_to_zero_pos) {
                std::cout << "from waiting to rotate_to_zero_pos\n";
                robot_state->test_phase = TestPhase::ROATE_TO_ZERO_POS;
                test_phase_trans_data->last_phase_end_time = frame_ptr->getGlobalTime();
                test_phase_trans_data->last_phase_end_robot_state = *robot_state;

                robot_cmd.is_rotate_to_zero_pos = false;
                robot_cmd.is_stop = false;
            } else if (robot_cmd.is_only_exec_torque_cmd) {
                std::cout << "from waiting to only_exec_torque_cmd\n";
                robot_state->test_phase = TestPhase::EXEC_TORQUE_CMD;
                test_phase_trans_data->last_phase_end_time = frame_ptr->getGlobalTime();
                test_phase_trans_data->last_phase_end_robot_state = *robot_state;

                robot_cmd.is_only_exec_torque_cmd = false;
                robot_cmd.is_stop = false;
            }
        }
        break;

        case TestPhase::ROATE_TO_NONZERO_POS: {
            const double target_angle = user_param->NonZeroAngle;

            Vector<double, 4> ref_left_arm_joint_pos, ref_right_arm_joint_pos,
                    ref_left_arm_joint_vel, ref_right_arm_joint_vel;

            Vector<double, 6> ref_left_leg_joint_pos, ref_right_leg_joint_pos,
                    ref_left_leg_joint_vel, ref_right_leg_joint_vel;

            for (int i = 0; i < 4; ++i) {
                interpolate_.cubic_spline_pos_and_vel(test_phase_trans_data->last_phase_end_robot_state.left_arm_joint_pos[i], target_angle, 
                    test_phase_trans_data->last_phase_end_robot_state.left_arm_joint_vel[i], 0, 
                    time_from_last_phase, 
                    user_param->InerpolateTime, 
                    user_param->MainThreadPeriod,
                    &ref_left_arm_joint_pos[i], &ref_left_arm_joint_vel[i]);

                interpolate_.cubic_spline_pos_and_vel(test_phase_trans_data->last_phase_end_robot_state.right_arm_joint_pos[i], target_angle, 
                    test_phase_trans_data->last_phase_end_robot_state.right_arm_joint_vel[i], 0, 
                    time_from_last_phase, 
                    user_param->InerpolateTime, 
                    user_param->MainThreadPeriod,
                    &ref_right_arm_joint_pos[i], &ref_right_arm_joint_vel[i]);
            }
            for (int i = 0; i < 6; ++i) {
                interpolate_.cubic_spline_pos_and_vel(test_phase_trans_data->last_phase_end_robot_state.left_leg_joint_pos[i], target_angle, 
                    test_phase_trans_data->last_phase_end_robot_state.left_leg_joint_vel[i], 0, 
                    time_from_last_phase, 
                    user_param->InerpolateTime, 
                    user_param->MainThreadPeriod,
                    &ref_left_leg_joint_pos[i], &ref_left_leg_joint_vel[i]);

                interpolate_.cubic_spline_pos_and_vel(test_phase_trans_data->last_phase_end_robot_state.right_leg_joint_pos[i], target_angle, 
                    test_phase_trans_data->last_phase_end_robot_state.right_leg_joint_vel[i], 0, 
                    time_from_last_phase, 
                    user_param->InerpolateTime, 
                    user_param->MainThreadPeriod,
                    &ref_right_leg_joint_pos[i], &ref_right_leg_joint_vel[i]);
            }

            auto& left_arm_joint_poscmd = frame_ptr->mutableJointCommand()->left_arm_joint_poscmd;
            auto& right_arm_joint_poscmd = frame_ptr->mutableJointCommand()->right_arm_joint_poscmd;
            auto& left_arm_joint_velcmd = frame_ptr->mutableJointCommand()->left_arm_joint_velcmd;
            auto& right_arm_joint_velcmd = frame_ptr->mutableJointCommand()->right_arm_joint_velcmd;
            auto& left_arm_joint_torcmd = frame_ptr->mutableJointCommand()->left_arm_joint_torcmd;
            auto& right_arm_joint_torcmd = frame_ptr->mutableJointCommand()->right_arm_joint_torcmd;
            auto& left_arm_joint_kp = frame_ptr->mutableJointCommand()->left_arm_joint_kp;
            auto& right_arm_joint_kp = frame_ptr->mutableJointCommand()->right_arm_joint_kp;
            auto& left_arm_joint_kd = frame_ptr->mutableJointCommand()->left_arm_joint_kd;
            auto& right_arm_joint_kd = frame_ptr->mutableJointCommand()->right_arm_joint_kd;

            auto& left_leg_joint_poscmd = frame_ptr->mutableJointCommand()->left_leg_joint_poscmd;
            auto& right_leg_joint_poscmd = frame_ptr->mutableJointCommand()->right_leg_joint_poscmd;            
            auto& left_leg_joint_velcmd = frame_ptr->mutableJointCommand()->left_leg_joint_velcmd;
            auto& right_leg_joint_velcmd = frame_ptr->mutableJointCommand()->right_leg_joint_velcmd;
            auto& left_leg_joint_torcmd = frame_ptr->mutableJointCommand()->left_leg_joint_torcmd;
            auto& right_leg_joint_torcmd = frame_ptr->mutableJointCommand()->right_leg_joint_torcmd;
            auto& left_leg_joint_kp = frame_ptr->mutableJointCommand()->left_leg_joint_kp;
            auto& right_leg_joint_kp = frame_ptr->mutableJointCommand()->right_leg_joint_kp;
            auto& left_leg_joint_kd = frame_ptr->mutableJointCommand()->left_leg_joint_kd;
            auto& right_leg_joint_kd = frame_ptr->mutableJointCommand()->right_leg_joint_kd;

            left_arm_joint_poscmd = ref_left_arm_joint_pos;
            right_arm_joint_poscmd = ref_right_arm_joint_pos;
            left_arm_joint_velcmd = ref_left_arm_joint_vel;
            right_arm_joint_velcmd = ref_right_arm_joint_vel;
            left_arm_joint_kp << user_param->Kp, user_param->Kp, user_param->Kp, user_param->Kp;
            right_arm_joint_kp << user_param->Kp, user_param->Kp, user_param->Kp, user_param->Kp;
            left_arm_joint_kd << user_param->Kd, user_param->Kd, user_param->Kd, user_param->Kd;
            right_arm_joint_kd << user_param->Kd, user_param->Kd, user_param->Kd, user_param->Kd;
            left_arm_joint_torcmd.setZero();
            right_arm_joint_torcmd.setZero();

            left_leg_joint_poscmd = ref_left_leg_joint_pos;
            right_leg_joint_poscmd = ref_right_leg_joint_pos;
            left_leg_joint_velcmd = ref_left_leg_joint_vel;
            right_leg_joint_velcmd = ref_right_leg_joint_vel;
            left_leg_joint_kp << user_param->Kp, 
                                 user_param->Kp, 
                                user_param->Kp, 
                                user_param->Kp,
			        user_param->Kp,	
                                user_param->Kp;
            right_leg_joint_kp << user_param->Kp, 
                                user_param->Kp, 
                                user_param->Kp, 
                                user_param->Kp, 
                                user_param->Kp,
				user_param->Kp;
            left_leg_joint_kd << user_param->Kd, 
                                 user_param->Kd, 
                                user_param->Kd, 
                                user_param->Kd,
			        user_param->Kd,	
                                user_param->Kd;
            right_leg_joint_kd << user_param->Kd, 
                                user_param->Kd, 
                                user_param->Kd, 
                                user_param->Kd, 
				user_param->Kd,
                                user_param->Kd;
            left_leg_joint_torcmd.setZero();
            right_leg_joint_torcmd.setZero();

            if (time_from_last_phase > user_param->InerpolateTime) {
                std::cout << "from rotate_to_nonzero_pos to waiting\n";
                robot_state->test_phase = TestPhase::WAITING;
                test_phase_trans_data->last_phase_end_time = frame_ptr->getGlobalTime();
                test_phase_trans_data->last_phase_end_robot_state = *robot_state;

                robot_cmd.is_stop = true;
            }
        }
        break;

        case TestPhase::ROATE_TO_ZERO_POS: {
            const double target_angle = 0.0;

            Vector<double, 4> ref_left_arm_joint_pos, ref_right_arm_joint_pos,
                    ref_left_arm_joint_vel, ref_right_arm_joint_vel;

            Vector<double, 6> ref_left_leg_joint_pos, ref_right_leg_joint_pos,
                    ref_left_leg_joint_vel, ref_right_leg_joint_vel;

            for (int i = 0; i < 4; ++i) {
                interpolate_.cubic_spline_pos_and_vel(test_phase_trans_data->last_phase_end_robot_state.left_arm_joint_pos[i], target_angle, 
                    test_phase_trans_data->last_phase_end_robot_state.left_arm_joint_vel[i], 0, 
                    time_from_last_phase, 
                    user_param->InerpolateTime, 
                    user_param->MainThreadPeriod,
                    &ref_left_arm_joint_pos[i], &ref_left_arm_joint_vel[i]);

                interpolate_.cubic_spline_pos_and_vel(test_phase_trans_data->last_phase_end_robot_state.right_arm_joint_pos[i], target_angle, 
                    test_phase_trans_data->last_phase_end_robot_state.right_arm_joint_vel[i], 0, 
                    time_from_last_phase, 
                    user_param->InerpolateTime, 
                    user_param->MainThreadPeriod,
                    &ref_right_arm_joint_pos[i], &ref_right_arm_joint_vel[i]);
            }
            for (int i = 0; i < 6; ++i) {
                interpolate_.cubic_spline_pos_and_vel(test_phase_trans_data->last_phase_end_robot_state.left_leg_joint_pos[i], target_angle, 
                    test_phase_trans_data->last_phase_end_robot_state.left_leg_joint_vel[i], 0, 
                    time_from_last_phase, 
                    user_param->InerpolateTime, 
                    user_param->MainThreadPeriod,
                    &ref_left_leg_joint_pos[i], &ref_left_leg_joint_vel[i]);

                interpolate_.cubic_spline_pos_and_vel(test_phase_trans_data->last_phase_end_robot_state.right_leg_joint_pos[i], target_angle, 
                    test_phase_trans_data->last_phase_end_robot_state.right_leg_joint_vel[i], 0, 
                    time_from_last_phase, 
                    user_param->InerpolateTime, 
                    user_param->MainThreadPeriod,
                    &ref_right_leg_joint_pos[i], &ref_right_leg_joint_vel[i]);
            }

            auto& left_arm_joint_poscmd = frame_ptr->mutableJointCommand()->left_arm_joint_poscmd;
            auto& right_arm_joint_poscmd = frame_ptr->mutableJointCommand()->right_arm_joint_poscmd;
            auto& left_arm_joint_velcmd = frame_ptr->mutableJointCommand()->left_arm_joint_velcmd;
            auto& right_arm_joint_velcmd = frame_ptr->mutableJointCommand()->right_arm_joint_velcmd;
            auto& left_arm_joint_torcmd = frame_ptr->mutableJointCommand()->left_arm_joint_torcmd;
            auto& right_arm_joint_torcmd = frame_ptr->mutableJointCommand()->right_arm_joint_torcmd;
            auto& left_arm_joint_kp = frame_ptr->mutableJointCommand()->left_arm_joint_kp;
            auto& right_arm_joint_kp = frame_ptr->mutableJointCommand()->right_arm_joint_kp;
            auto& left_arm_joint_kd = frame_ptr->mutableJointCommand()->left_arm_joint_kd;
            auto& right_arm_joint_kd = frame_ptr->mutableJointCommand()->right_arm_joint_kd;

            auto& left_leg_joint_poscmd = frame_ptr->mutableJointCommand()->left_leg_joint_poscmd;
            auto& right_leg_joint_poscmd = frame_ptr->mutableJointCommand()->right_leg_joint_poscmd;            
            auto& left_leg_joint_velcmd = frame_ptr->mutableJointCommand()->left_leg_joint_velcmd;
            auto& right_leg_joint_velcmd = frame_ptr->mutableJointCommand()->right_leg_joint_velcmd;
            auto& left_leg_joint_torcmd = frame_ptr->mutableJointCommand()->left_leg_joint_torcmd;
            auto& right_leg_joint_torcmd = frame_ptr->mutableJointCommand()->right_leg_joint_torcmd;
            auto& left_leg_joint_kp = frame_ptr->mutableJointCommand()->left_leg_joint_kp;
            auto& right_leg_joint_kp = frame_ptr->mutableJointCommand()->right_leg_joint_kp;
            auto& left_leg_joint_kd = frame_ptr->mutableJointCommand()->left_leg_joint_kd;
            auto& right_leg_joint_kd = frame_ptr->mutableJointCommand()->right_leg_joint_kd;

            left_arm_joint_poscmd = ref_left_arm_joint_pos;
            right_arm_joint_poscmd = ref_right_arm_joint_pos;
            left_arm_joint_velcmd = ref_left_arm_joint_vel;
            right_arm_joint_velcmd = ref_right_arm_joint_vel;
            left_arm_joint_kp << user_param->Kp, user_param->Kp, user_param->Kp, user_param->Kp;
            right_arm_joint_kp << user_param->Kp, user_param->Kp, user_param->Kp, user_param->Kp;
            left_arm_joint_kd << user_param->Kd, user_param->Kd, user_param->Kd, user_param->Kd;
            right_arm_joint_kd << user_param->Kd, user_param->Kd, user_param->Kd, user_param->Kd;
            left_arm_joint_torcmd.setZero();
            right_arm_joint_torcmd.setZero();

            left_leg_joint_poscmd = ref_left_leg_joint_pos;
            right_leg_joint_poscmd = ref_right_leg_joint_pos;
            left_leg_joint_velcmd = ref_left_leg_joint_vel;
            right_leg_joint_velcmd = ref_right_leg_joint_vel;
            left_leg_joint_kp << user_param->Kp, 
                                user_param->Kp, 
                                user_param->Kp, 
                                user_param->Kp,
				                user_param->Kp, 
                                user_param->Kp;
            right_leg_joint_kp << user_param->Kp, 
                                user_param->Kp, 
                                user_param->Kp, 
                                user_param->Kp, 
                                user_param->Kp,
				                user_param->Kp;
            left_leg_joint_kd << user_param->Kd, 
                                user_param->Kd, 
                                user_param->Kd, 
                               	user_param->Kd,
			       	            user_param->Kd, 
                                user_param->Kd;
            right_leg_joint_kd << user_param->Kd, 
                                user_param->Kd, 
                                user_param->Kd,
				                user_param->Kd, 
                                user_param->Kd, 
                                user_param->Kd;
            left_leg_joint_torcmd.setZero();
            right_leg_joint_torcmd.setZero();

            if (time_from_last_phase > user_param->InerpolateTime) {
                std::cout << "from rotate_to_zero_pos to waiting\n";
                robot_state->test_phase = TestPhase::WAITING;
                test_phase_trans_data->last_phase_end_time = frame_ptr->getGlobalTime();
                test_phase_trans_data->last_phase_end_robot_state = *robot_state;

                robot_cmd.is_stop = true;
            }
        }
        break;

        case TestPhase::EXEC_TORQUE_CMD: {
            const double target_tor = user_param->TorCmdValue;

            auto& left_arm_joint_poscmd = frame_ptr->mutableJointCommand()->left_arm_joint_poscmd;
            auto& right_arm_joint_poscmd = frame_ptr->mutableJointCommand()->right_arm_joint_poscmd;
            auto& left_arm_joint_velcmd = frame_ptr->mutableJointCommand()->left_arm_joint_velcmd;
            auto& right_arm_joint_velcmd = frame_ptr->mutableJointCommand()->right_arm_joint_velcmd;
            auto& left_arm_joint_torcmd = frame_ptr->mutableJointCommand()->left_arm_joint_torcmd;
            auto& right_arm_joint_torcmd = frame_ptr->mutableJointCommand()->right_arm_joint_torcmd;
            auto& left_arm_joint_kp = frame_ptr->mutableJointCommand()->left_arm_joint_kp;
            auto& right_arm_joint_kp = frame_ptr->mutableJointCommand()->right_arm_joint_kp;
            auto& left_arm_joint_kd = frame_ptr->mutableJointCommand()->left_arm_joint_kd;
            auto& right_arm_joint_kd = frame_ptr->mutableJointCommand()->right_arm_joint_kd;

            auto& left_leg_joint_poscmd = frame_ptr->mutableJointCommand()->left_leg_joint_poscmd;
            auto& right_leg_joint_poscmd = frame_ptr->mutableJointCommand()->right_leg_joint_poscmd;            
            auto& left_leg_joint_velcmd = frame_ptr->mutableJointCommand()->left_leg_joint_velcmd;
            auto& right_leg_joint_velcmd = frame_ptr->mutableJointCommand()->right_leg_joint_velcmd;
            auto& left_leg_joint_torcmd = frame_ptr->mutableJointCommand()->left_leg_joint_torcmd;
            auto& right_leg_joint_torcmd = frame_ptr->mutableJointCommand()->right_leg_joint_torcmd;
            auto& left_leg_joint_kp = frame_ptr->mutableJointCommand()->left_leg_joint_kp;
            auto& right_leg_joint_kp = frame_ptr->mutableJointCommand()->right_leg_joint_kp;
            auto& left_leg_joint_kd = frame_ptr->mutableJointCommand()->left_leg_joint_kd;
            auto& right_leg_joint_kd = frame_ptr->mutableJointCommand()->right_leg_joint_kd;

            left_arm_joint_poscmd.setZero();
            right_arm_joint_poscmd.setZero();
            left_arm_joint_velcmd.setZero();
            right_arm_joint_velcmd.setZero();
            left_arm_joint_kp.setZero();
            right_arm_joint_kp.setZero();
            left_arm_joint_kd.setZero();
            right_arm_joint_kd.setZero();
            left_arm_joint_torcmd << target_tor, target_tor, target_tor, target_tor;
            right_arm_joint_torcmd << target_tor, target_tor, target_tor, target_tor;

            left_leg_joint_poscmd.setZero();
            right_leg_joint_poscmd.setZero();
            left_leg_joint_velcmd.setZero();
            right_leg_joint_velcmd.setZero();
            left_leg_joint_kp.setZero();
            right_leg_joint_kp.setZero();
            left_leg_joint_kd.setZero();
            right_leg_joint_kd.setZero();
            left_leg_joint_torcmd << target_tor, target_tor, target_tor, target_tor, target_tor;
            right_leg_joint_torcmd << target_tor, target_tor, target_tor, target_tor, target_tor;

            if (time_from_last_phase > user_param->TorExecTime) {
                std::cout << "from exec_tor_cmd to waiting\n";
                robot_state->test_phase = TestPhase::WAITING;
                test_phase_trans_data->last_phase_end_time = frame_ptr->getGlobalTime();
                test_phase_trans_data->last_phase_end_robot_state = *robot_state;

                robot_cmd.is_stop = true;
            }
        }
        break;
}
}

void FSM_State_Test::onExit() {

}

TransitionData FSM_State_Test::transition() {
    this->transition_data_.done = true;

    return this->transition_data_;
}

FSM_StateName FSM_State_Test::checkTransition() {
    //default no transition
    this->next_state_name_ = this->state_name_;
    //iter++;
    
    return this->next_state_name_;
}
