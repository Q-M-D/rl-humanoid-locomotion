//
// Created by han on 2021/6/15.
//

#ifndef AMBER_ETHERCAT_RT_ETHERCAT_H
#define AMBER_ETHERCAT_RT_ETHERCAT_H

#include "cstdint"

// for abs
#include <unordered_map>
#include <memory>
#include "EthercatSlaveBase.h"
#include "ImuRc.h"


extern std::unordered_map<int, std::shared_ptr<EthercatSlaveBase>> slave_dict;

void rt_ethercat_init(std::string if_name);
//void rt_ethercat_config();
void rt_ethercat_run();

void rt_ethercat_get_data();
void rt_ethercat_set_command();


#endif //AMBER_ETHERCAT_RT_ETHERCAT_H
