// //
// // Created by han on 23-9-3.
// //

// #include "can_protocol_func.h"

// #define KP_MIN 0.0f
// #define KP_MAX 500.0f
// #define KD_MIN 0.0f
// #define KD_MAX 5.0f
// #define POS_MIN (-12.5f)
// #define POS_MAX 12.5f
// #define SPD_MIN (-18.0f)
// #define SPD_MAX 18.0f
// #define T_MIN (-30.0f)
// #define T_MAX 30.0f
// #define I_MIN (-30.0f)
// #define I_MAX 30.0f

// void can_msg_hs_unpack(const std::shared_ptr<ControlFSMData>& a, int slave_no, uint64_t data)
// {
//     uint64_t tmp_ff = 0xff;
//     for (int i = 0; i < 6; ++i) {
//         auto id_tmp = (uint8_t)((data >> ((i) * 8)) & tmp_ff);
//         a->hs_data[slave_no][i] = id_tmp;
//     }
// }

// int float_to_uint(float x, float x_min, float x_max, int bits) {
//     float span = x_max - x_min;
//     float offset = x_min;
//     return (int)((x - offset) * ((float)((1 << bits) - 1)) / span);
// }

// float uint_to_float(int x_int, float x_min, float x_max, int bits) {
//     float span = x_max - x_min;
//     float offset = x_min;
//     return ((float)x_int) * span / ((float)((1 << bits) - 1)) + offset;
// }


// void pack_pvt_cmd(uint8_t *buffer, float kp, float kd, float pos, float spd, float tor)
// {
//     int kp_int;
//     int kd_int;
//     int pos_int;
//     int spd_int;
//     int tor_int;

//     if (kp > KP_MAX)
//         kp = KP_MAX;
//     else if (kp < KP_MIN)
//         kp = KP_MIN;
//     if (kd > KD_MAX)
//         kd = KD_MAX;
//     else if (kd < KD_MIN)
//         kd = KD_MIN;
//     if (pos > POS_MAX)
//         pos = POS_MAX;
//     else if (pos < POS_MIN)
//         pos = POS_MIN;
//     if (spd > SPD_MAX)
//         spd = SPD_MAX;
//     else if (spd < SPD_MIN)
//         spd = SPD_MIN;
//     if (tor > T_MAX)
//         tor = T_MAX;
//     else if (tor < T_MIN)
//         tor = T_MIN;

//     kp_int = float_to_uint(kp, KP_MIN, KP_MAX, 12);
//     kd_int = float_to_uint(kd, KD_MIN, KD_MAX, 9);
//     pos_int = float_to_uint(pos, POS_MIN, POS_MAX, 16);
//     spd_int = float_to_uint(spd, SPD_MIN, SPD_MAX, 12);
//     tor_int = float_to_uint(tor, T_MIN, T_MAX, 12);

//     buffer[0] = 0x00 | (kp_int >> 7);                             // kp5
//     buffer[1] = ((kp_int & 0x7F) << 1) | ((kd_int & 0x100) >> 8); // kp7+kd1
//     buffer[2] = kd_int & 0xFF;
//     buffer[3] = pos_int >> 8;
//     buffer[4] = pos_int & 0xFF;
//     buffer[5] = spd_int >> 4;
//     buffer[6] = (spd_int & 0x0F) << 4 | (tor_int >> 8);
//     buffer[7] = tor_int & 0xff;
// }

// void unpack_pvt_data(uint8_t *buffer, const std::shared_ptr<ControlFSMData>& a, int node_no, int motor_no)
// {
//     int pos_int = buffer[1] << 8 | buffer[2];
//     int spd_int = buffer[3] << 4 | (buffer[4] & 0xF0) >> 4;
//     int cur_int = (buffer[4] & 0x0F) << 8 | buffer[5];

//     a->position_act_raw_[node_no][motor_no] = uint_to_float(pos_int, POS_MIN, POS_MAX, 16);
//     a->velocity_act_raw_[node_no][motor_no] = uint_to_float(spd_int, SPD_MIN, SPD_MAX, 12);
//     a->current_act_raw_[node_no][motor_no] = uint_to_float(cur_int, I_MIN, I_MAX, 12);
//     a->temperature_raw_[node_no][motor_no] = (buffer[6] - 50.0) / 2.0;
//     a->mos_tmp_raw_[node_no][motor_no] = (buffer[7] - 50.0) / 2.0;
// }

// /**
//  *这个函数名为 unpack_pvt_data，它的作用是解析从设备传来的数据，并将解析后的值存储到一个名为 a 的 ControlFSMData 类型的对象中。
//  具体来说，这个函数接收一个指向 uint8_t 类型的缓冲区的指针 buffer，以及一个指向 ControlFSMData 类型的共享指针 a。除此之外，它还接收了几个整数参数 node_no、motor_no 和 joint_no。

//     在函数内部，它首先根据协议解析了缓冲区中的数据，并将解析后的值存储到 a 对象的各个成员变量中，这些成员变量包括位置、速度、电流、温度等。
//     这些值经过一些特定的转换和计算，最终被存储到 ControlFSMData 对象中，以便后续的控制逻辑可以使用这些数据进行决策和控制。
// */
// void unpack_pvt_data(uint8_t *buffer, const std::shared_ptr<ControlFSMData>& a, int node_no, int motor_no, int joint_no)
// {
//     int pos_int = buffer[1] << 8 | buffer[2];
//     int spd_int = buffer[3] << 4 | (buffer[4] & 0xF0) >> 4;
//     int cur_int = (buffer[4] & 0x0F) << 8 | buffer[5];

//     a->position_act_raw_[node_no][motor_no] = a->joint_dir_[joint_no]*uint_to_float(pos_int, POS_MIN, POS_MAX, 16);
//     a->velocity_act_raw_[node_no][motor_no] = a->joint_dir_[joint_no]*uint_to_float(spd_int, SPD_MIN, SPD_MAX, 12);
//     a->current_act_raw_[node_no][motor_no] = a->joint_dir_[joint_no]*uint_to_float(cur_int, static_cast<float>(a->joint_min_current_[joint_no]),
//                                                            static_cast<float>(a->joint_max_current_[joint_no]), 12);
//     a->temperature_raw_[node_no][motor_no] = (buffer[6] - 50.0) / 2.0;
//     a->mos_tmp_raw_[node_no][motor_no] = (buffer[7] - 50.0) / 2.0;
// }

// // todo: assume rang is symmetry
// void pack_pvt_cmd(uint8_t *buffer, float kp, float kd, float pos, float spd, float tor, int joint_no, const std::shared_ptr<ControlFSMData>& a)
// {
//     int kp_int;
//     int kd_int;
//     int pos_int;
//     int spd_int;
//     int tor_int;

//     if (kp > KP_MAX)
//         kp = KP_MAX;
//     else if (kp < KP_MIN)
//         kp = KP_MIN;
//     if (kd > KD_MAX)
//         kd = KD_MAX;
//     else if (kd < KD_MIN)
//         kd = KD_MIN;
//     if (pos > POS_MAX)
//         pos = POS_MAX;
//     else if (pos < POS_MIN)
//         pos = POS_MIN;
//     if (spd > SPD_MAX)
//         spd = SPD_MAX;
//     else if (spd < SPD_MIN)
//         spd = SPD_MIN;
//     if (tor > a->joint_max_torque_[joint_no])
//         tor = static_cast<float>(a->joint_max_torque_[joint_no]);
//     else if (tor < a->joint_min_torque_[joint_no])
//         tor = static_cast<float>(a->joint_min_torque_[joint_no]);

//     float new_pos = (float)a->joint_dir_[joint_no]*pos;
//     float new_spd = (float)a->joint_dir_[joint_no]*spd;
//     float new_tor = (float)a->joint_dir_[joint_no]*tor;
//     kp_int = float_to_uint(kp, KP_MIN, KP_MAX, 12);
//     kd_int = float_to_uint(kd, KD_MIN, KD_MAX, 9);
//     pos_int = float_to_uint(new_pos, POS_MIN, POS_MAX, 16);
//     spd_int = float_to_uint(new_spd, SPD_MIN, SPD_MAX, 12);
//     tor_int = float_to_uint(new_tor, static_cast<float>(a->joint_min_torque_[joint_no]), static_cast<float>(a->joint_max_torque_[joint_no]), 12);

//     buffer[0] = 0x00 | (kp_int >> 7);                             // kp5
//     buffer[1] = ((kp_int & 0x7F) << 1) | ((kd_int & 0x100) >> 8); // kp7+kd1
//     buffer[2] = kd_int & 0xFF;
//     buffer[3] = pos_int >> 8;
//     buffer[4] = pos_int & 0xFF;
//     buffer[5] = spd_int >> 4;
//     buffer[6] = (spd_int & 0x0F) << 4 | (tor_int >> 8);
//     buffer[7] = tor_int & 0xff;
// }

// void read_motor_setting(const EthercatOptionsFromYaml& config, const std::shared_ptr<ControlFSMData>& a)
// {
//     // right arm(0:3) -> left arm(4:7) -> right leg(8:12) -> left leg(13:17)
//     a->joint_max_current_[0] = config.motor_max_current[config.right_arm_motor_type[0]];
//     a->joint_max_current_[1] = config.motor_max_current[config.right_arm_motor_type[1]];
//     a->joint_max_current_[2] = config.motor_max_current[config.right_arm_motor_type[2]];
//     a->joint_max_current_[3] = config.motor_max_current[config.right_arm_motor_type[3]];

//     a->joint_max_current_[4] = config.motor_max_current[config.left_arm_motor_type[0]];
//     a->joint_max_current_[5] = config.motor_max_current[config.left_arm_motor_type[1]];
//     a->joint_max_current_[6] = config.motor_max_current[config.left_arm_motor_type[2]];
//     a->joint_max_current_[7] = config.motor_max_current[config.left_arm_motor_type[3]];

//     a->joint_max_current_[8] = config.motor_max_current[config.right_leg_motor_type[0]];
//     a->joint_max_current_[9] = config.motor_max_current[config.right_leg_motor_type[1]];
//     a->joint_max_current_[10] = config.motor_max_current[config.right_leg_motor_type[2]];
//     a->joint_max_current_[11] = config.motor_max_current[config.right_leg_motor_type[3]];
//     a->joint_max_current_[12] = config.motor_max_current[config.right_leg_motor_type[4]];

//     a->joint_max_current_[13] = config.motor_max_current[config.left_leg_motor_type[0]];
//     a->joint_max_current_[14] = config.motor_max_current[config.left_leg_motor_type[1]];
//     a->joint_max_current_[15] = config.motor_max_current[config.left_leg_motor_type[2]];
//     a->joint_max_current_[16] = config.motor_max_current[config.left_leg_motor_type[3]];
//     a->joint_max_current_[17] = config.motor_max_current[config.left_leg_motor_type[4]];

//     a->joint_min_current_[0] = config.motor_min_current[config.right_arm_motor_type[0]];
//     a->joint_min_current_[1] = config.motor_min_current[config.right_arm_motor_type[1]];
//     a->joint_min_current_[2] = config.motor_min_current[config.right_arm_motor_type[2]];
//     a->joint_min_current_[3] = config.motor_min_current[config.right_arm_motor_type[3]];

//     a->joint_min_current_[4] = config.motor_min_current[config.left_arm_motor_type[0]];
//     a->joint_min_current_[5] = config.motor_min_current[config.left_arm_motor_type[1]];
//     a->joint_min_current_[6] = config.motor_min_current[config.left_arm_motor_type[2]];
//     a->joint_min_current_[7] = config.motor_min_current[config.left_arm_motor_type[3]];

//     a->joint_min_current_[8] = config.motor_min_current[config.right_leg_motor_type[0]];
//     a->joint_min_current_[9] = config.motor_min_current[config.right_leg_motor_type[1]];
//     a->joint_min_current_[10] = config.motor_min_current[config.right_leg_motor_type[2]];
//     a->joint_min_current_[11] = config.motor_min_current[config.right_leg_motor_type[3]];
//     a->joint_min_current_[12] = config.motor_min_current[config.right_leg_motor_type[4]];

//     a->joint_min_current_[13] = config.motor_min_current[config.left_leg_motor_type[0]];
//     a->joint_min_current_[14] = config.motor_min_current[config.left_leg_motor_type[1]];
//     a->joint_min_current_[15] = config.motor_min_current[config.left_leg_motor_type[2]];
//     a->joint_min_current_[16] = config.motor_min_current[config.left_leg_motor_type[3]];
//     a->joint_min_current_[17] = config.motor_min_current[config.left_leg_motor_type[4]];


//     a->joint_max_torque_[0] = config.motor_max_torque[config.right_arm_motor_type[0]];
//     a->joint_max_torque_[1] = config.motor_max_torque[config.right_arm_motor_type[1]];
//     a->joint_max_torque_[2] = config.motor_max_torque[config.right_arm_motor_type[2]];
//     a->joint_max_torque_[3] = config.motor_max_torque[config.right_arm_motor_type[3]];

//     a->joint_max_torque_[4] = config.motor_max_torque[config.left_arm_motor_type[0]];
//     a->joint_max_torque_[5] = config.motor_max_torque[config.left_arm_motor_type[1]];
//     a->joint_max_torque_[6] = config.motor_max_torque[config.left_arm_motor_type[2]];
//     a->joint_max_torque_[7] = config.motor_max_torque[config.left_arm_motor_type[3]];

//     a->joint_max_torque_[8] = config.motor_max_torque[config.right_leg_motor_type[0]];
//     a->joint_max_torque_[9] = config.motor_max_torque[config.right_leg_motor_type[1]];
//     a->joint_max_torque_[10] = config.motor_max_torque[config.right_leg_motor_type[2]];
//     a->joint_max_torque_[11] = config.motor_max_torque[config.right_leg_motor_type[3]];
//     a->joint_max_torque_[12] = config.motor_max_torque[config.right_leg_motor_type[4]];

//     a->joint_max_torque_[13] = config.motor_max_torque[config.left_leg_motor_type[0]];
//     a->joint_max_torque_[14] = config.motor_max_torque[config.left_leg_motor_type[1]];
//     a->joint_max_torque_[15] = config.motor_max_torque[config.left_leg_motor_type[2]];
//     a->joint_max_torque_[16] = config.motor_max_torque[config.left_leg_motor_type[3]];
//     a->joint_max_torque_[17] = config.motor_max_torque[config.left_leg_motor_type[4]];

//     a->joint_min_torque_[0] = config.motor_min_torque[config.right_arm_motor_type[0]];
//     a->joint_min_torque_[1] = config.motor_min_torque[config.right_arm_motor_type[1]];
//     a->joint_min_torque_[2] = config.motor_min_torque[config.right_arm_motor_type[2]];
//     a->joint_min_torque_[3] = config.motor_min_torque[config.right_arm_motor_type[3]];

//     a->joint_min_torque_[4] = config.motor_min_torque[config.left_arm_motor_type[0]];
//     a->joint_min_torque_[5] = config.motor_min_torque[config.left_arm_motor_type[1]];
//     a->joint_min_torque_[6] = config.motor_min_torque[config.left_arm_motor_type[2]];
//     a->joint_min_torque_[7] = config.motor_min_torque[config.left_arm_motor_type[3]];

//     a->joint_min_torque_[8] = config.motor_min_torque[config.right_leg_motor_type[0]];
//     a->joint_min_torque_[9] = config.motor_min_torque[config.right_leg_motor_type[1]];
//     a->joint_min_torque_[10] = config.motor_min_torque[config.right_leg_motor_type[2]];
//     a->joint_min_torque_[11] = config.motor_min_torque[config.right_leg_motor_type[3]];
//     a->joint_min_torque_[12] = config.motor_min_torque[config.right_leg_motor_type[4]];

//     a->joint_min_torque_[13] = config.motor_min_torque[config.left_leg_motor_type[0]];
//     a->joint_min_torque_[14] = config.motor_min_torque[config.left_leg_motor_type[1]];
//     a->joint_min_torque_[15] = config.motor_min_torque[config.left_leg_motor_type[2]];
//     a->joint_min_torque_[16] = config.motor_min_torque[config.left_leg_motor_type[3]];
//     a->joint_min_torque_[17] = config.motor_min_torque[config.left_leg_motor_type[4]];

//     //for (int i = 0; i < 5; ++i) {
//         //a->left_leg_direction_[i] = config.left_leg_motor_dir[i];
//         //a->right_leg_direction_[i] = config.right_leg_motor_dir[i];
//    // }

//     // right arm(0:3) -> left arm(4:7) -> right leg(8:12) -> left leg(13:17)
//     a->joint_dir_[0] = config.right_leg_motor_dir[0];
//     a->joint_dir_[1] = config.right_leg_motor_dir[1];
//     a->joint_dir_[2] = config.right_leg_motor_dir[2];
//     a->joint_dir_[3] = config.right_leg_motor_dir[3];

//     a->joint_dir_[4] = config.left_leg_motor_dir[0];
//     a->joint_dir_[5] = config.left_leg_motor_dir[1];
//     a->joint_dir_[6] = config.left_leg_motor_dir[2];
//     a->joint_dir_[7] = config.left_leg_motor_dir[3];



//     a->joint_dir_[8] = config.right_leg_motor_dir[0];
//     a->joint_dir_[9] = config.right_leg_motor_dir[1];
//     a->joint_dir_[10] = config.right_leg_motor_dir[2];
//     a->joint_dir_[11] = config.right_leg_motor_dir[3];
//     a->joint_dir_[12] = config.right_leg_motor_dir[4];

//     a->joint_dir_[13] = config.left_leg_motor_dir[0];
//     a->joint_dir_[14] = config.left_leg_motor_dir[1];
//     a->joint_dir_[15] = config.left_leg_motor_dir[2];
//     a->joint_dir_[16] = config.left_leg_motor_dir[3];
//     a->joint_dir_[17] = config.left_leg_motor_dir[4];


//     //TODO: config
//     for(int i = 0; i < 4; ++i) {
//         for (int j = 0; j < 6; ++j) {
//             a->joint_offset_[i][j] = 0;
//         }
//     }
// }

//
// Created by han on 23-9-3.
//

#include "can_protocol_func.h"

double temp_current_pos_left_motor3 = 0.0;
double temp_current_pos_left_motor4 = 0.0;
double temp_current_pos_right_motor3 = 0.0;
double temp_current_pos_right_motor4 = 0.0;

void check_usingOffset_ornot(const std::shared_ptr<ControlFSMData>& a,double current_pos, int node_no, int motor_no)
{
    if(node_no==2)
    {
        if(motor_no==2)
        {
           //a->joint_offset_[node_no][motor_no] = 6.28;
           if((current_pos != -12.5) && (current_pos > 3.14) && (temp_current_pos_right_motor3 == -12.5))
           {
              a->joint_offset_[node_no][motor_no] = 6.28;
              std::cout << "***************************************************************" << std::endl;
              std::cout << "current adjustment motor is right leg no.3 motor,offset is 6.28" << std::endl;
           }
           temp_current_pos_right_motor3 = current_pos;
        }
        else if(motor_no == 3)
        {
           if((current_pos != -12.5) && (current_pos > 3.14) && (temp_current_pos_right_motor4 == -12.5))
           {
                a->joint_offset_[node_no][motor_no] = 6.28;
                std::cout << "***************************************************************" << std::endl;
                std::cout << "current adjustment motor  is right leg no.4 motor,offset is 6.28" << std::endl;
           }
           temp_current_pos_right_motor4 = current_pos;
        }
    }
    else if(node_no==3)
    {
        if(motor_no==2)
        {
           if((current_pos != -12.5) && (current_pos > 3.14) && (temp_current_pos_left_motor3 == -12.5))
           {
              a->joint_offset_[node_no][motor_no] = 6.28;
              std::cout << "***************************************************************" << std::endl;
              std::cout << "current adjustment motor  is left leg no.3 motor,offset is 6.28" << std::endl;
           }
           temp_current_pos_left_motor3 = current_pos;
        }
        else if(motor_no == 3)
        {
           if((current_pos != -12.5) && (current_pos > 3.14) && (temp_current_pos_left_motor4 == -12.5))
           {
              a->joint_offset_[node_no][motor_no] = 6.28;
              std::cout << "***************************************************************" << std::endl;
              std::cout << "current adjustment motor  is left leg no.4 motor,offset is 6.28" << std::endl;
           }
           temp_current_pos_left_motor4 = current_pos;
        }
    }
}
int float_to_uint_ex(float x, float x_min, float x_max, int bits) {
    float span = x_max - x_min;
    float offset = x_min;
    return (int) ((x - offset) * ((float) ((1 << bits) - 1)) / span);
}

float uint_to_float_ex(int x_int, float x_min, float x_max, int bits) {
    float span = x_max - x_min;
    float offset = x_min;
    return ((float)x_int) * span / ((float)((1 << bits) - 1)) + offset;
}


void unpack_pvt_data_ex(uint8_t *buffer, const std::shared_ptr<ControlFSMData>& a, int node_no, int motor_no, int joint_no, const MotorType& motor_type)
{
    switch (motor_type) {
        case MotorType::DMBOT: 
        case MotorType::INKEXBOT:{
	    int error = (buffer[0] >> 4) & 0x0F;
            int pos_int = buffer[1] << 8 | buffer[2];
            int spd_int = buffer[3] << 4 | (buffer[4] & 0xF0) >> 4;
            int cur_int = (buffer[4] & 0x0F) << 8 | buffer[5];

            a->position_act_raw_[node_no][motor_no] = a->joint_dir_[joint_no]*uint_to_float_ex(
                    pos_int, static_cast<float>(a->joint_min_position_[joint_no]),
                    static_cast<float>(a->joint_max_position_[joint_no]), 16);

            a->velocity_act_raw_[node_no][motor_no] = a->joint_dir_[joint_no]*uint_to_float_ex(
                    spd_int, static_cast<float>(a->joint_min_velocity_[joint_no]),
                    static_cast<float>(a->joint_max_velocity_[joint_no]), 12);

            a->torque_act_raw_[node_no][motor_no] = a->joint_dir_[joint_no]*uint_to_float_ex(cur_int, static_cast<float>(a->joint_min_torque_[joint_no]),
                                                                                            static_cast<float>(a->joint_max_torque_[joint_no]), 12);
            a->joint_error[node_no][motor_no] = error;
	}
        break;
        case MotorType::LZBOT: {
            int pos_int = buffer[0] << 8 | buffer[1];
            int spd_int = buffer[2] << 8 | buffer[3];
            int cur_int = buffer[4] << 8 | buffer[5];

            a->position_act_raw_[node_no][motor_no] = a->joint_dir_[joint_no]*uint_to_float_ex(
                    pos_int, static_cast<float>(a->joint_min_position_[joint_no]),
                    static_cast<float>(a->joint_max_position_[joint_no]), 16);
		
	        check_usingOffset_ornot(a,a->position_act_raw_[node_no][motor_no],node_no,motor_no);

            a->velocity_act_raw_[node_no][motor_no] = a->joint_dir_[joint_no]*uint_to_float_ex(
                    spd_int, static_cast<float>(a->joint_min_velocity_[joint_no]),
                    static_cast<float>(a->joint_max_velocity_[joint_no]), 16);

            a->torque_act_raw_[node_no][motor_no] = a->joint_dir_[joint_no]*uint_to_float_ex(cur_int, static_cast<float>(a->joint_min_torque_[joint_no]),
                                                                                      static_cast<float>(a->joint_max_torque_[joint_no]), 16);
        }
        break;
        default:
        break;
    }
    

    /*a->temperature_raw_[node_no][motor_no] = (buffer[6] - 50.0) / 2.0;
    a->mos_tmp_raw_[node_no][motor_no] = (buffer[7] - 50.0) / 2.0;*/
}

// todo: assume rang is symmetry
int pack_pvt_cmd_ex(uint8_t *buffer, float kp, float kd, float pos, float spd, float tor, int joint_no, const std::shared_ptr<ControlFSMData>& a, const MotorType& motor_type)
{
    int kp_int;
    int kd_int;
    int pos_int;
    int spd_int;
    int tor_int;

    if (kp > 500)
        kp = 500;
    else if (kp < 0)
        kp = 0;
    if (kd > 5)
        kd = 5;
    else if (kd < 0)
        kd = 0;
    if (pos > static_cast<float>(a->joint_max_position_[joint_no]))
        pos = static_cast<float>(a->joint_max_position_[joint_no]);
    else if (pos < static_cast<float>(a->joint_min_position_[joint_no]))
        pos = static_cast<float>(a->joint_min_position_[joint_no]);
    if (spd > static_cast<float>(a->joint_max_velocity_[joint_no]))
        spd = static_cast<float>(a->joint_max_velocity_[joint_no]);
    else if (spd < static_cast<float>(a->joint_min_velocity_[joint_no]))
        spd = static_cast<float>(a->joint_min_velocity_[joint_no]);
    if (tor > a->joint_max_torque_[joint_no])
        tor = static_cast<float>(a->joint_max_torque_[joint_no]);
    else if (tor < a->joint_min_torque_[joint_no])
        tor = static_cast<float>(a->joint_min_torque_[joint_no]);

    float new_pos = (float)a->joint_dir_[joint_no]*pos;
    float new_spd = (float)a->joint_dir_[joint_no]*spd;
    float new_tor = (float)a->joint_dir_[joint_no]*tor;

    switch (motor_type) {
        case MotorType::DMBOT: {
            kp_int = float_to_uint_ex(kp, 0, 500, 12);
            kd_int = float_to_uint_ex(kd, 0, 5, 9);
            pos_int = float_to_uint_ex(new_pos,
                                    static_cast<float>(a->joint_min_position_[joint_no]),
                                    static_cast<float>(a->joint_max_position_[joint_no]), 16);
            spd_int = float_to_uint_ex(new_spd, static_cast<float>(a->joint_min_velocity_[joint_no]),
                                    static_cast<float>(a->joint_max_velocity_[joint_no]), 12);
            tor_int = float_to_uint_ex(new_tor, static_cast<float>(a->joint_min_torque_[joint_no]), static_cast<float>(a->joint_max_torque_[joint_no]), 12);

            buffer[0] = pos_int >> 8;
            buffer[1] = pos_int;
            buffer[2] = spd_int >> 4;
            buffer[3] = ((spd_int & 0xF)<<4)|(kp_int >> 8);
            buffer[4] = kp_int;
            buffer[5] = (kd_int >> 4);
            buffer[6] = ((kd_int&0xF)<<4)|(tor_int>>8);
            buffer[7] = tor_int;
        }
        break;
        case MotorType::LZBOT: {
            kp_int = float_to_uint_ex(kp, 0, 500, 16);
            kd_int = float_to_uint_ex(kd, 0, 5, 16);
            pos_int = float_to_uint_ex(new_pos,
                                    static_cast<float>(a->joint_min_position_[joint_no]),
                                    static_cast<float>(a->joint_max_position_[joint_no]), 16);
            spd_int = float_to_uint_ex(new_spd, static_cast<float>(a->joint_min_velocity_[joint_no]),
                                    static_cast<float>(a->joint_max_velocity_[joint_no]), 16);
            tor_int = float_to_uint_ex(new_tor, static_cast<float>(a->joint_min_torque_[joint_no]), static_cast<float>(a->joint_max_torque_[joint_no]), 16);

            buffer[0] = pos_int >> 8;
            buffer[1] = pos_int;
            buffer[2] = spd_int >> 8;
            buffer[3] = spd_int;
            buffer[4] = kp_int >> 8;
            buffer[5] = kp_int;
            buffer[6] = kd_int >> 8;
            buffer[7] = kd_int;
        }
        break;
        case MotorType::INKEXBOT: {
            kp_int = float_to_uint_ex(kp, 0, 500, 12);
            kd_int = float_to_uint_ex(kd, 0, 5, 9);

            pos_int = float_to_uint_ex(new_pos,
                                    static_cast<float>(a->joint_min_position_[joint_no]),
                                    static_cast<float>(a->joint_max_position_[joint_no]), 16);
            spd_int = float_to_uint_ex(new_spd, static_cast<float>(a->joint_min_velocity_[joint_no]),
                                    static_cast<float>(a->joint_max_velocity_[joint_no]), 12);
            tor_int = float_to_uint_ex(new_tor, static_cast<float>(a->joint_min_torque_[joint_no]), static_cast<float>(a->joint_max_torque_[joint_no]), 12);

            buffer[0] = 0x00 | (kp_int >> 7);                             // kp5
            buffer[1] = ((kp_int & 0x7F) << 1) | ((kd_int & 0x100) >> 8); // kp7+kd1
            buffer[2] = kd_int & 0xFF;
            buffer[3] = pos_int >> 8;
            buffer[4] = pos_int & 0xFF;
            buffer[5] = spd_int >> 4;
            buffer[6] = (spd_int & 0x0F) << 4 | (tor_int >> 8);
            buffer[7] = tor_int & 0xff;
        }
        break;
        default:
        break;
    }
    return tor_int;
}

void calibrate_cur_pos_as_zero_pos(uint8_t *buffer, const MotorType& motor_type) {
    switch (motor_type) {
        case MotorType::YOBOT: {
            buffer[0] = 0xFF;
            buffer[1] = 0xFF;
            buffer[2] = 0xFF;
            buffer[3] = 0xFF;
            buffer[4] = 0xFF;
            buffer[5] = 0xFF;
            buffer[6] = 0xF1;
            buffer[7] = 0x00;
        }
        break;

        case MotorType::DMBOT: {
            buffer[0] = 0xFF;
            buffer[1] = 0xFF;
            buffer[2] = 0xFF;
            buffer[3] = 0xFF;
            buffer[4] = 0xFF;
            buffer[5] = 0xFF;
            buffer[6] = 0xFF;
            buffer[7] = 0xFE;
        }
        break;
    }
}
