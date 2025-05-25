#include "record_log.h"

void add_data(std::ofstream& outfile, const VectorXd& data, const int& data_num) {
    for (int i = 0; i < data_num; ++i) {
        outfile << data[i] << ",";
    }
}

bool record_debug_data(const std::shared_ptr<Frame>& frame_ptr, 
    const std::string& path) {
    if (frame_ptr == nullptr) {
        std::cout << "[log_file]meet nullptr, skip record debug data" << std::endl;
        return false;
    }
    const auto& robot_state = frame_ptr->mutableRobotState();
    const auto& joint_cmd = frame_ptr->mutableJointCommand();


    std::ofstream out_file;
    out_file.open(path, std::ios::app);

    //global time
    out_file << frame_ptr->getGlobalTime() << ",";

    //joint actual pos
    add_data(out_file, robot_state->left_leg_joint_pos, 6);
    add_data(out_file, robot_state->right_leg_joint_pos, 6);

    //joint actual vel
    add_data(out_file, robot_state->left_leg_joint_vel, 6);
    add_data(out_file, robot_state->right_leg_joint_vel, 6);

    //joint actual tor
    add_data(out_file, robot_state->left_leg_joint_tor, 6);
    add_data(out_file, robot_state->right_leg_joint_tor, 6);

    out_file << static_cast<int>(robot_state->test_phase) << ",";

    //joint cmd pos\vel\tor
    add_data(out_file, joint_cmd->left_leg_joint_poscmd, 6);
    add_data(out_file, joint_cmd->right_leg_joint_poscmd, 6);
    add_data(out_file, joint_cmd->left_leg_joint_velcmd, 6);
    add_data(out_file, joint_cmd->right_leg_joint_velcmd, 6);
    add_data(out_file, joint_cmd->left_leg_joint_torcmd, 6);
    add_data(out_file, joint_cmd->right_leg_joint_torcmd, 6);

    add_data(out_file, joint_cmd->left_arm_joint_poscmd, 4);
    add_data(out_file, joint_cmd->right_arm_joint_poscmd, 4);
    add_data(out_file, joint_cmd->left_arm_joint_velcmd, 4);
    add_data(out_file, joint_cmd->right_arm_joint_velcmd, 4);
    add_data(out_file, joint_cmd->left_arm_joint_torcmd, 4);
    add_data(out_file, joint_cmd->right_arm_joint_torcmd, 4);

    add_data(out_file, robot_state->left_arm_joint_pos, 4);
    add_data(out_file, robot_state->right_arm_joint_pos, 4);
    add_data(out_file, robot_state->left_arm_joint_vel, 4);
    add_data(out_file, robot_state->right_arm_joint_vel, 4);
    add_data(out_file, robot_state->left_arm_joint_tor, 4);
    add_data(out_file, robot_state->right_arm_joint_tor, 4);

    out_file << 0;
    out_file << std::endl;
    out_file.close();

    return true;
}