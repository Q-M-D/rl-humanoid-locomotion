//
// Created by han on 23-9-3.
//

#ifndef BAER_CORE_CAN_PROTOCOL_FUNC_H
#define BAER_CORE_CAN_PROTOCOL_FUNC_H

#include "cstdint"
#include "../FSM_States/ControlFSMData.h"
#include "EthercatParameter.h"

enum class MotorType {
    YOBOT = 1, DMBOT, LZBOT, INKEXBOT, UNKNOWN
};

// void can_msg_hs_unpack(const std::shared_ptr<ControlFSMData>& a, int slave_no, uint64_t data);

// int float_to_uint(float x, float x_min, float x_max, int bits);
// void pack_pvt_cmd(uint8_t *buffer, float kp, float kd, float pos, float spd, float tor);
// void pack_pvt_cmd(uint8_t *buffer, float kp, float kd, float pos, float spd, float tor, int joint_no, const std::shared_ptr<ControlFSMData>& a);
// void unpack_pvt_data(uint8_t *buffer, const std::shared_ptr<ControlFSMData>& a, int node_no, int motor_no);
// void unpack_pvt_data(uint8_t *buffer, const std::shared_ptr<ControlFSMData>& a, int node_no, int motor_no, int joint_no);


void read_motor_setting(const EthercatOptionsFromYaml& config, const std::shared_ptr<ControlFSMData>& a);

int float_to_uint_ex(float x, float x_min, float x_max, int bits);

int pack_pvt_cmd_ex(uint8_t *buffer, float kp, float kd, float pos, float spd, float tor,
                  int joint_no, const std::shared_ptr<ControlFSMData>& a, const MotorType& motor_type);

void unpack_pvt_data_ex(uint8_t *buffer, const std::shared_ptr<ControlFSMData>& a,
                     int node_no, int motor_no, int joint_no, const MotorType& motor_type);

void calibrate_cur_pos_as_zero_pos(uint8_t *buffer, const MotorType& motor_type);


#endif //BAER_CORE_CAN_PROTOCOL_FUNC_H
