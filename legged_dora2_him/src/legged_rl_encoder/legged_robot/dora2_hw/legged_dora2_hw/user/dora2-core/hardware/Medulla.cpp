//
// Created by han on 9/20/23.
//

#include "Medulla.h"
#include "string.h"


Medulla::Medulla(const std::string& name, const uint32_t address):
        EthercatSlaveBase(name, address){}

void Medulla::updateRead()
{
    memcpy(&medulla_data_, ec_slave[address_].inputs, sizeof(MedullaData));
}

void Medulla::updateWrite()
{
    memcpy(ec_slave[address_].outputs, &medulla_cmd_, sizeof(MedullaCommand));
}