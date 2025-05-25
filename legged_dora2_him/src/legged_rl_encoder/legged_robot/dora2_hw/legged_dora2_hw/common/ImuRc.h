//
// Created by han on 23-4-4.
//

#ifndef BAER_ETHERCAT_IMURC_H
#define BAER_ETHERCAT_IMURC_H

#include "EthercatSlaveBase.h"
#include "ethercat.h"

struct ImuRcData{
    float       acc_x;
    float       acc_y;
    float       acc_z;
    float       gyr_x;
    float       gyr_y;
    float       gyr_z;
    float       q0;
    float       q1;
    float       q2;
    float       q3;
    uint32_t    imu_status_word;
    uint32_t    rc_status;
    uint16_t    rc_ch2;
    uint16_t    rc_ch3;
    uint16_t    rc_ch4;
    uint16_t    rc_ch5;
    uint16_t    rc_ch6;
    uint16_t    rc_ch7;
    uint16_t    rc_ch8;
    uint16_t    rc_ch9;
    uint16_t    rc_ch10;
    uint16_t    rc_ch11;
    uint16_t    rc_ch12;
    uint16_t    rc_ch13;
    uint16_t    rc_ch14;
    uint16_t    rc_ch15;
    uint16_t    rc_ch16;
    uint16_t    rc_ch1;
};

struct ImuRcCommand{
    uint32_t STO;
    uint32_t power;
};


class ImuRc : public EthercatSlaveBase{
public:
    ImuRc(const std::string& name, uint32_t address);

    ImuRcData imu_rc_data_;
    ImuRcCommand imu_rc_cmd_;

    void updateRead() override;
    void updateWrite() override;

};


#endif //BAER_ETHERCAT_IMURC_H
