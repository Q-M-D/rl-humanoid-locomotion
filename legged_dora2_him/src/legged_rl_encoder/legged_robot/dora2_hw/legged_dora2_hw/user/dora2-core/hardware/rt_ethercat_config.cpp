// //
// // Created by han on 23-9-3.
// //

// #include "rt_ethercat_config.h"

// #include "string"
// #include "memory"
// #include "Medulla.h"

// //TODO: config
// void rt_ethercat_config()
// {
//     std::string name_1("left");
//     std::shared_ptr<EthercatSlaveBase> tmp = std::make_shared<Medulla>(name_1, 1);
//     slave_dict[0] = tmp;

//     std::string name_2("right");
//     std::shared_ptr<EthercatSlaveBase> tmp_2 = std::make_shared<Medulla>(name_2, 2);
//     slave_dict[1] = tmp_2;

//     std::string name_3("right_arm");
//     std::shared_ptr<EthercatSlaveBase> tmp_3 = std::make_shared<Medulla>(name_3, 3);
//     slave_dict[2] = tmp_3;

//     std::string name_4("left_arm");
//     std::shared_ptr<EthercatSlaveBase> tmp_4 = std::make_shared<Medulla>(name_4, 4);
//     slave_dict[3] = tmp_4;

//     std::string name_5("imu_rc");
//     std::shared_ptr<EthercatSlaveBase> tmp_5 = std::make_shared<ImuRc>(name_5, 5);
//     slave_dict[4] = tmp_5;
// }

//
// Created by han on 23-9-3.
//

#include "rt_ethercat_config.h"

#include "string"
#include "memory"
#include "Medulla.h"

//TODO: config
void rt_ethercat_config()
{
    std::string name_1("first_mcu");
    std::shared_ptr<EthercatSlaveBase> tmp = std::make_shared<Medulla>(name_1, 1);
    slave_dict[0] = tmp;

    std::string name_2("second_mcu");
    std::shared_ptr<EthercatSlaveBase> tmp_2 = std::make_shared<Medulla>(name_2, 2);
    slave_dict[1] = tmp_2;

    std::string name_3("third_mcu");
    std::shared_ptr<EthercatSlaveBase> tmp_3 = std::make_shared<Medulla>(name_3, 3);
    slave_dict[2] = tmp_3;

    std::string name_4("imu_rc");
    std::shared_ptr<EthercatSlaveBase> tmp_4 = std::make_shared<ImuRc>(name_4, 4);
    slave_dict[3] = tmp_4;
}
