//
// Created by han on 23-4-4.
//

#include "ImuRc.h"
#include "cstring"

ImuRc::ImuRc(const std::string& name, uint32_t address):
        EthercatSlaveBase(name, address){}

void ImuRc::updateRead()
{
    memcpy(&imu_rc_data_, ec_slave[address_].inputs, sizeof(ImuRcData));
}

void ImuRc::updateWrite()
{
    memcpy(ec_slave[address_].outputs, &imu_rc_cmd_, sizeof(ImuRcCommand));
}
