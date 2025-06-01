#include "dora2HW.h"

double global_time = 0.0;
// Medulla* node_1;
// Medulla* node_2;
// Medulla* node_3;
// Medulla* node_4;
// ImuRc* imu_rc;

namespace legged
{

  bool dora2HW::init(ros::NodeHandle &root_nh, ros::NodeHandle &robot_hw_nh)
  { 
    //发布话题
    motorPosPublisher_ = robot_hw_nh.advertise<std_msgs::Float64MultiArray>("data_analysis/motor_pos", 1);
    motorVelPublisher_ = robot_hw_nh.advertise<std_msgs::Float64MultiArray>("data_analysis/motor_vel", 1);
    motorTorquePublisher_ = robot_hw_nh.advertise<std_msgs::Float64MultiArray>("data_analysis/motor_torque", 1);
    motor_pos_feedback_.setZero();
    motor_vel_feedback_.setZero();
    motor_tau_feedback_.setZero();
    joint_planned_torque_.setZero();

    //std::string net_card;
    std::string user_param_path;
    std::string ethercat_param_path;
    // 通过rosparam读取EtherCAT配置选项

/*     // root_nh.getParam("/ethercat_config/net_card", config.net_card);
    // root_nh.getParam("/ethercat_config/ctr_freq", config.ctr_freq);
    // root_nh.getParam("/ethercat_config/robot_type", config.robot_type);
    // root_nh.getParam("/ethercat_config/node_motor_num", config.node_motor_num);
    // root_nh.getParam("/ethercat_config/motor_max_torque", config.motor_max_torque);
    // root_nh.getParam("/ethercat_config/motor_min_torque", config.motor_min_torque);
    // root_nh.getParam("/ethercat_config/motor_max_current", config.motor_max_current);
    // root_nh.getParam("/ethercat_config/motor_min_current", config.motor_min_current);
    // root_nh.getParam("/ethercat_config/right_arm_motor_type", config.right_arm_motor_type);
    // root_nh.getParam("/ethercat_config/left_arm_motor_type", config.left_arm_motor_type);
    // root_nh.getParam("/ethercat_config/right_leg_motor_type", config.right_leg_motor_type);
    // root_nh.getParam("/ethercat_config/left_leg_motor_type", config.left_leg_motor_type);
    // root_nh.getParam("/ethercat_config/right_leg_motor_dir", config.right_leg_motor_dir);
    // root_nh.getParam("/ethercat_config/left_leg_motor_dir", config.left_leg_motor_dir);
    // root_nh.getParam("/ethercat_config/right_arm_motor_dir", config.right_arm_motor_dir);
    // root_nh.getParam("/ethercat_config/left_arm_motor_dir", config.left_arm_motor_dir);
    // std::cout<<"motor_max_current:"<<config.motor_min_current[0]<<std::endl;

    // if (!root_nh.getParam("/ethercat_cofig/net_card", net_card)) {
    //     net_card = "enp3s0";
    //     ROS_WARN("Parameter depth_image_topic not set. Using default: %s", net_card.c_str());
    // }
    // 通过rosparam读取config.yaml路径 */

    //drake 可能用不了 可以请教杨太文有没有其他解决方案，现在是在.h文件中直接读取的
    if (!root_nh.getParam("/user_param_path", user_param_path)) {
        user_param_path = "/home/mmlab/legged_dora2_erect/src/legged_rl_encoder/legged_robot/dora2_hw/legged_dora2_hw/config/user_param_path.yaml";
        ROS_WARN("Parameter user_param_path not set. Using default: %s", user_param_path.c_str());
    }
    if (!root_nh.getParam("/ethercat_param_path", ethercat_param_path)) {
        ethercat_param_path = "/home/mmlab/legged_dora2_erect/src/legged_rl_encoder/legged_robot/dora2_hw/legged_dora2_hw/config/ethercat_config.yaml";
        ROS_WARN("Parameter ethercat_param_path not set. Using default: %s", ethercat_param_path.c_str());
    }

    // 从YAML文件加载EtherCAT配置选项
    // auto ethercat_options_from_yaml = drake::yaml::LoadYamlFile<EthercatOptionsFromYaml>(ethercat_param_path);
    
    // std::cout<<"user_param_path:"<<user_param_path<<std::endl;

    // 创建用户参数对象并从YAML文件初始化
    std::shared_ptr<RemoteUserParameter> user_param = std::make_shared<RemoteUserParameter>();
    try {
        user_param->initializeFromYamlFile(user_param_path);
    } catch(std::exception& e) {
        printf("Failed to initialize robot parameters from yaml file: %s\n", e.what());
    }
   
    // 创建机器人模型对象
    robot_model = std::make_shared<RobotModel>();
    std::cout<<"robot_model "<<robot_model<<std::endl;
    
    // 创建帧对象，包含机器人模型和用户参数 
    frame_ptr = std::make_shared<Frame>(robot_model, user_param);
    std::cout<<"frame_ptr "<<frame_ptr<<std::endl;
    
    // 创建控制状态机数据对象并初始化
    control_fsm_data = std::make_shared<ControlFSMData>(frame_ptr);
    std::cout<<"control_fsm_data "<<control_fsm_data<<std::endl;
    // 通过yaml读取电机设置
    // motor_setting(control_fsm_data);
    // 读取电机设置并配置EtherCAT
    //read_motor_setting(ethercat_options_from_yaml, control_fsm_data);
    /**
    这段代码是一个名为 rt_ethercat_config 的函数，主要用于配置以太网通信中的从设备。在这段代码中，它创建了几个不同的从设备（EthercatSlaveBase），
    并将它们添加到一个名为 slave_dict 的字典中。每个从设备都有一个名称和一个编号，通过 std::make_shared 创建了一个指向该从设备的共享指针，并将其添加到 slave_dict 中。
    总体来说，这段代码的作用是初始化和配置多个从设备，使它们准备好进行以太网通信。
    */
    rt_ethercat_config();

    // 设置控制状态机数据节点
    // control_fsm_data->node_1_ = slave_dict[0];
    // control_fsm_data->node_2_ = slave_dict[1];
    // control_fsm_data->node_3_ = slave_dict[2];
    // control_fsm_data->node_4_ = slave_dict[3];

    control_fsm_data->node_1_ = slave_dict[0];
    control_fsm_data->node_2_ = slave_dict[1];
    control_fsm_data->node_3_ = slave_dict[2];

    // 初始化EtherCAT接口
    // std::string if_name(ethercat_options_from_yaml.net_card);
    // rt_ethercat_init(if_name);
    
    //可以写死
    
    printf("aaaaaaaaaaaaaaaaaaaaa");
    std::string if_name = "enxc8a3625563bc";
    rt_ethercat_init(if_name);
    printf("llllllllllllllllllllll");
  

    listener_ = new tf::TransformListener(root_nh, ros::Duration(5.0), true);

    if (!LeggedHW::init(root_nh, robot_hw_nh))
    {
      printf("flase_bc_leggedHW::init\n");
      return false;
    }
    setupJoints(); //urdf需要随之变化
    setupImu();
    //setupContactSensor(robot_hw_nh);
    return true;
  }
  template <int row_>

  using Vector = Eigen::Matrix<double, row_, 1>;

  Vector<12> m_q; // motor feedback (prior conversion)
  Vector<12> m_v; // motor feedback (prior conversion)
  Vector<12> m_t; // motor feedback (prior conversion)

  void dora2HW::read(const ros::Time &time, const ros::Duration &period)
  { 
    // 将从slave_dict获取的设备转换为Medulla和IMU_RC类型
    node_1 = dynamic_cast<Medulla*>(slave_dict[0].get());
    node_2 = dynamic_cast<Medulla*>(slave_dict[1].get());
    node_3 = dynamic_cast<Medulla*>(slave_dict[2].get());
    //node_4 = dynamic_cast<Medulla*>(slave_dict[3].get());
    imu_rc = dynamic_cast<ImuRc*>(slave_dict[3].get());
    // 增加心跳信号
    ++hs;
    // 更新每个medulla节点的控制字和心跳信号
    control_fsm_data->control_word_ = 1;
    node_1->medulla_cmd_.hs = hs;
    node_1->medulla_cmd_.control_word = control_fsm_data->control_word_;
    node_2->medulla_cmd_.hs = hs;
    node_2->medulla_cmd_.control_word = control_fsm_data->control_word_;
    node_3->medulla_cmd_.hs = hs;
    node_3->medulla_cmd_.control_word = control_fsm_data->control_word_;
    // node_4->medulla_cmd_.hs = hs;
    // node_4->medulla_cmd_.control_word = control_fsm_data->control_word_;
    // 通过etherCAT协议获取数据
    rt_ethercat_get_data();
    // unpack motor data
    // left leg in node_1
    // 1.left leg
    // 1. hip yaw
    //  解析从设备传来的数据，并将解析后的值存储到一个名为 control_fsm_data 的 ControlFSMData 类型的对象中。
        // 复制潜在变量
    if (control_fsm_data == nullptr)
    {   std::cout<<"control_fsm_data "<<control_fsm_data<<std::endl;
        std::cout << "control_fsm_data is null" << std::endl;
    }
    //新的手和脚read如下，按照需要删减
    //1) node_1 left arm + right arm

 //1. left shoulder pitch
    byte_8.udata = node_1->medulla_data_.motor_1;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 1, 0, 4, MotorType::DMBOT);

    //2. left shoulder roll
    byte_8.udata = node_1->medulla_data_.motor_2;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 1, 1, 5, MotorType::DMBOT);

    //3. left shoulder yaw
    byte_8.udata = node_1->medulla_data_.motor_3;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 1, 2, 6, MotorType::DMBOT);

    //4. left elbow pitch
    byte_8.udata = node_1->medulla_data_.motor_4;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 1, 3, 7, MotorType::DMBOT);

    //5. right shoulder pitch
    byte_8.udata = node_1->medulla_data_.motor_5;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 0, 0, 0, MotorType::DMBOT);

    //2) node_2 left leg + right arm

    //1. left hip yaw
    byte_8.udata = node_2->medulla_data_.motor_1;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 3, 0, 14, MotorType::INKEXBOT);

    //2. left hip roll
    byte_8.udata = node_2->medulla_data_.motor_2;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 3, 1, 15, MotorType::INKEXBOT);

    //3. left hip pitch
    byte_8.udata = node_2->medulla_data_.motor_3;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 3, 2, 16, MotorType::INKEXBOT);
    //control_fsm_data->joint_error[3][2] = node_2->medulla_data_.lz_3_error;

    //4. left knee
    byte_8.udata = node_2->medulla_data_.motor_4;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 3, 3, 17, MotorType::INKEXBOT);
    //control_fsm_data->joint_error[3][3] = node_2->medulla_data_.lz_4_error;

    //5. left toe
    byte_8.udata = node_2->medulla_data_.motor_5;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 3, 4, 18, MotorType::INKEXBOT);

    //6. left toe
    byte_8.udata = node_2->medulla_data_.motor_6;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 3, 5, 19, MotorType::INKEXBOT);

    //7. right shoulder roll
    byte_8.udata = node_2->medulla_data_.motor_7;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 0, 1, 1, MotorType::DMBOT);

    //8. right shoulder yaw
    byte_8.udata = node_2->medulla_data_.motor_8;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 0, 2, 2, MotorType::DMBOT);

    //9. right elbow pitch
    byte_8.udata = node_2->medulla_data_.motor_9;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 0, 3, 3, MotorType::DMBOT);

    //3) node_3 right leg
    //1. right hip yaw
    byte_8.udata = node_3->medulla_data_.motor_1;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 2, 0, 8, MotorType::INKEXBOT);

    //1. right hip roll
    byte_8.udata = node_3->medulla_data_.motor_2;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 2, 1, 9, MotorType::INKEXBOT);

    //3. right hip pitch
    byte_8.udata = node_3->medulla_data_.motor_3;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 2, 2, 10, MotorType::INKEXBOT);
    control_fsm_data->joint_error[2][2] = node_3->medulla_data_.lz_3_error;

    //4. right knee
    byte_8.udata = node_3->medulla_data_.motor_4;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 2, 3, 11, MotorType::INKEXBOT);
    control_fsm_data->joint_error[2][3] = node_3->medulla_data_.lz_4_error;

    //5. right toe
    byte_8.udata = node_3->medulla_data_.motor_5;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 2, 4, 12, MotorType::INKEXBOT);

    //6. right toe
    byte_8.udata = node_3->medulla_data_.motor_6;
    unpack_pvt_data_ex(byte_8.buffer, control_fsm_data, 2, 5, 13, MotorType::INKEXBOT);
    // byte_8.udata = node_1->medulla_data_.motor_1;
    // unpack_pvt_data(byte_8.buffer, control_fsm_data, 3, 0, 13);
    // byte_8.udata = node_1->medulla_data_.motor_2;
    // // 2. hip roll
    // unpack_pvt_data(byte_8.buffer, control_fsm_data, 3, 1, 14);
    // byte_8.udata = node_1->medulla_data_.motor_4;
    // // 3. hip pitch
    // unpack_pvt_data(byte_8.buffer, control_fsm_data, 3, 2, 15);
    // byte_8.udata = node_1->medulla_data_.motor_5;
    // //4. knee
    // unpack_pvt_data(byte_8.buffer, control_fsm_data, 3, 3, 16);
    // byte_8.udata = node_1->medulla_data_.motor_6;
    // //5. toe
    // unpack_pvt_data(byte_8.buffer, control_fsm_data, 3, 4, 17);

    // //2.right leg
    // // 1. hip yaw
    // byte_8.udata = node_2->medulla_data_.motor_1;
    // unpack_pvt_data(byte_8.buffer, control_fsm_data, 2, 0, 8);
    // byte_8.udata = node_2->medulla_data_.motor_2;
    // // 2. hip roll
    // unpack_pvt_data(byte_8.buffer, control_fsm_data, 2, 1, 9);
    // byte_8.udata = node_2->medulla_data_.motor_4;
    // // 3. hip pitch
    // unpack_pvt_data(byte_8.buffer, control_fsm_data, 2, 2, 10);
    // byte_8.udata = node_2->medulla_data_.motor_5;
    // //4. knee
    // unpack_pvt_data(byte_8.buffer, control_fsm_data, 2, 3, 11);
    // byte_8.udata = node_2->medulla_data_.motor_6;
    // //5. toe
    // unpack_pvt_data(byte_8.buffer, control_fsm_data, 2, 4, 12);


/*     //3. right arm
    // 1. pitch
    unpack_pvt_data(byte_8.buffer, control_fsm_data, 0, 0, 0);

    // 2. roll
    unpack_pvt_data(byte_8.buffer, control_fsm_data, 0, 1, 1);

    // 3. yaw
    unpack_pvt_data(byte_8.buffer, control_fsm_data, 0, 2, 2);

    //4. pitch
    unpack_pvt_data(byte_8.buffer, control_fsm_data, 0, 3, 3);

    //4. left arm
    // 1. pitch
    unpack_pvt_data(byte_8.buffer, control_fsm_data, 1, 0, 4);

    // 2. roll
    unpack_pvt_data(byte_8.buffer, control_fsm_data, 1, 1, 5);

    // 3. yaw
    unpack_pvt_data(byte_8.buffer, control_fsm_data, 1, 2, 6);

    //4. pitch
    unpack_pvt_data(byte_8.buffer, control_fsm_data, 1, 3, 7); */
    
    // 获取全局时间并设置到帧中
    global_time = control_fsm_data->time_; 
    frame_ptr->setGlobalTime(global_time);

    /*
    元生艾欸姆尤
    */ 
    // lord IMU需要进行坐标值转换，可以请教董子扬
    CalibrateImu(false);

    // 更新机器人状态中的左腿关节数据
    for (int i = 0; i < 6; ++i) {
        joint_data_[i].pos_ = (control_fsm_data->position_act_raw_[3][i] - control_fsm_data->joint_offset_[3][i]) * direction_motor[i] + bias_motor[i];
        joint_data_[i].vel_  = control_fsm_data->velocity_act_raw_[3][i] * direction_motor[i];
        joint_data_[i].tau_  = control_fsm_data->torque_act_raw_[3][i] * direction_motor[i];
        motor_pos_feedback_(i) = joint_data_[i].pos_;
        motor_vel_feedback_(i) = joint_data_[i].vel_;
        motor_tau_feedback_(i) = joint_data_[i].tau_;
    }

    // 更新机器人状态中的右腿关节数据
    for (int i = 0; i < 6; ++i) {
        joint_data_[i+6].pos_ = (control_fsm_data->position_act_raw_[2][i] - control_fsm_data->joint_offset_[2][i]) * direction_motor[i+6] + bias_motor[i+6];
        joint_data_[i+6].vel_  = control_fsm_data->velocity_act_raw_[2][i] * direction_motor[i+6];
        joint_data_[i+6].tau_  = control_fsm_data->torque_act_raw_[2][i] * direction_motor[i+6];
        motor_pos_feedback_(i+6) = joint_data_[i+6].pos_;
        motor_vel_feedback_(i+6) = joint_data_[i+6].vel_;
        motor_tau_feedback_(i+6) = joint_data_[i+6].tau_;
    }

    motorTorquePublisher_.publish(createFloat64MultiArrayFromVector(motor_tau_feedback_));
    motorPosPublisher_.publish(createFloat64MultiArrayFromVector(motor_pos_feedback_));
    motorVelPublisher_.publish(createFloat64MultiArrayFromVector(motor_vel_feedback_));

/*     // 更新机器人状态中的左臂关节数据
    for (int i = 0; i < 4; ++i) {
        joint_data_[i+4].pos_ = (control_fsm_data->position_act_raw_[1][i] - control_fsm_data->joint_offset_[1][i]);
        joint_data_[i+4].vel_  = control_fsm_data->velocity_act_raw_[1][i];
        joint_data_[i+4].tau_  = control_fsm_data->torque_act_raw_[1][i];
    }

    // 更新机器人状态中的右臂关节数据
    for (int i = 0; i < 4; ++i) {
        joint_data_[i].pos_ = (control_fsm_data->position_act_raw_[0][i] - control_fsm_data->joint_offset_[0][i]);
        joint_data_[i].vel_ = control_fsm_data->velocity_act_raw_[0][i];
        joint_data_[i].tau_  = control_fsm_data->torque_act_raw_[0][i];
    } */

    // Set feedforward and velocity cmd to zero to avoid for safety when not controller setCommand
    std::vector<std::string> names = hybridJointInterface_.getNames();
    for (const auto &name : names)
    {
      HybridJointHandle handle = hybridJointInterface_.getHandle(name);
      handle.setFeedforward(0.);
      handle.setVelocityDesired(0.);
      handle.setKd(3.1415);
      handle.setKp(0.);
    }
    
  }


  void dora2HW::write(const ros::Time &time, const ros::Duration &period)
  {
    // 500HZ
    int temp_tor = 0;
    /*
      第一步 命令赋值
    */
    // left leg
    for (int i = 0; i < 6; ++i) {
        control_fsm_data->kp_[3][i] = joint_data_[i].kp_;
        control_fsm_data->kd_[3][i] = joint_data_[i].kd_;
        control_fsm_data->position_des_[3][i] = (joint_data_[i].pos_des_  - bias_motor[i]) * direction_motor[i];
        control_fsm_data->velocity_des_[3][i] = joint_data_[i].vel_des_ * direction_motor[i];
        // control_fsm_data->velocity_des_[3][i] = joint_data_[i].vel_des_ * direction_motor[i];
        control_fsm_data->torque_des_[3][i] = joint_data_[i].ff_;
        // control_fsm_data->torque_des_[3][i] = 0;
        
        std::cout << "left leg num" << i << ":  kp:" << control_fsm_data->kp_[3][i] << "  kd:" << control_fsm_data->kd_[3][i]
        << "  pos:" << control_fsm_data->position_des_[3][i] << " vel:" <<  control_fsm_data->velocity_des_[3][i] <<
        " tor:" <<  control_fsm_data->torque_des_[3][i] << std::endl;
    }
    //  right leg
    for (int i = 0; i < 6; ++i) {
        control_fsm_data->kp_[2][i] = joint_data_[i+6].kp_;
        control_fsm_data->kd_[2][i] = joint_data_[i+6].kd_;
        control_fsm_data->position_des_[2][i] = (joint_data_[i+6].pos_des_  - bias_motor[i+6]) * direction_motor[i+6];
        control_fsm_data->velocity_des_[2][i] = joint_data_[i+6].vel_des_ * direction_motor[i+6];
        control_fsm_data->torque_des_[2][i] = joint_data_[i+6].ff_;
        // control_fsm_data->torque_des_[2][i] = 0;

        std::cout << "right leg num" << i << ":  kp:" << control_fsm_data->kp_[2][i] << "  kd:" << control_fsm_data->kd_[2][i]
        << "  pos:" << control_fsm_data->position_des_[2][i] << " vel:" <<  control_fsm_data->velocity_des_[2][i] <<
        " tor:" <<  control_fsm_data->torque_des_[2][i] << std::endl;
    }
    // //  left arm
    // for (int i = 0; i < 4; ++i) {
    //     control_fsm_data->kp_[1][i] = joint_data_[i+4].kp_;
    //     control_fsm_data->kd_[1][i] = joint_data_[i+4].kd_;
    //     control_fsm_data->position_des_[1][i] = joint_data_[i+4].pos_des_;
    //     control_fsm_data->velocity_des_[1][i] = joint_data_[i+4].vel_des_;
    //     control_fsm_data->torque_des_[1][i] = joint_data_[i+4].ff_;
    // }
    // //  right arm
    // for (int i = 0; i < 4; ++i) {
    //     control_fsm_data->kp_[0][i] = joint_data_[i].kp_;
    //     control_fsm_data->kd_[0][i] = joint_data_[i].kd_;
    //     control_fsm_data->position_des_[0][i] = joint_data_[i].pos_des_;
    //     control_fsm_data->velocity_des_[0][i] = joint_data_[i].vel_des_;
    //     control_fsm_data->torque_des_[0][i] = joint_data_[i].ff_;
    // } 


    // left leg
    // for (int i = 0; i < 6; ++i) {
    //     control_fsm_data->kp_[3][i] = 0;
    //     control_fsm_data->kd_[3][i] = 0;
    //     control_fsm_data->position_des_[3][i] = 0;
    //     control_fsm_data->velocity_des_[3][i] = 0;
    //     control_fsm_data->torque_des_[3][i] = 0;
    // }
    // //  right leg
    // for (int i = 0; i < 6; ++i) {
    //     control_fsm_data->kp_[2][i] = 0;
    //     control_fsm_data->kd_[2][i] = 0;
    //     control_fsm_data->position_des_[2][i] = 0;
    //     control_fsm_data->velocity_des_[2][i] = 0;
    //     control_fsm_data->torque_des_[2][i] = 0;
    // }

    //  left arm
    for (int i = 0; i < 4; ++i) {
        control_fsm_data->kp_[1][i] = 0;
        control_fsm_data->kd_[1][i] = 3;
        control_fsm_data->position_des_[1][i] = 0;
        control_fsm_data->velocity_des_[1][i] = 0;
        control_fsm_data->torque_des_[1][i] = 0;
    }
    //  right arm
    for (int i = 0; i < 4; ++i) {
        control_fsm_data->kp_[0][i] = 0;
        control_fsm_data->kd_[0][i] = 3;
        control_fsm_data->position_des_[0][i] = 0;
        control_fsm_data->velocity_des_[0][i] = 0;
        control_fsm_data->torque_des_[0][i] = 0;
    }

    /*
      第二部 命令格式转换 将 control_fsm_data 数据存储到 byte_8.buffer
    */
    //新的手和脚write如下，按照需要删减
    //1) node_1 left arm + right arm

        //1. left shoulder pitch
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[1][0], (float)control_fsm_data->kd_[1][0],
                    (float)control_fsm_data->position_des_[1][0] + (float)control_fsm_data->joint_offset_[1][0], (float)control_fsm_data->velocity_des_[1][0],
                    (float)control_fsm_data->torque_des_[1][0], 4, control_fsm_data, MotorType::DMBOT);
        node_1->medulla_cmd_.motor_1 = byte_8.udata;
        //2. left shoulder roll
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[1][1], (float)control_fsm_data->kd_[1][1],
                    (float)control_fsm_data->position_des_[1][1] + (float)control_fsm_data->joint_offset_[1][1], (float)control_fsm_data->velocity_des_[1][1],
                    (float)control_fsm_data->torque_des_[1][1], 5, control_fsm_data, MotorType::DMBOT);
        node_1->medulla_cmd_.motor_2 = byte_8.udata;

        //3. left shoulder yaw
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[1][2], (float)control_fsm_data->kd_[1][2],
                    ((float)control_fsm_data->position_des_[1][2] + (float)control_fsm_data->joint_offset_[1][2]), (float)control_fsm_data->velocity_des_[1][2],
                    (float)control_fsm_data->torque_des_[1][2], 6, control_fsm_data, MotorType::DMBOT);
        node_1->medulla_cmd_.motor_3 = byte_8.udata;

        //4. left elbow pitch
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[1][3], (float)control_fsm_data->kd_[1][3],
                    ((float)control_fsm_data->position_des_[1][3]+ (float)control_fsm_data->joint_offset_[1][3]), (float)control_fsm_data->velocity_des_[1][3],
                    (float)control_fsm_data->torque_des_[1][3], 7, control_fsm_data, MotorType::DMBOT);
        node_1->medulla_cmd_.motor_4 = byte_8.udata;

        //5. right shoulder pitch
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[0][0], (float)control_fsm_data->kd_[0][0],
                    ((float)control_fsm_data->position_des_[0][0]+ (float)control_fsm_data->joint_offset_[0][0]), (float)control_fsm_data->velocity_des_[0][0],
                    (float)control_fsm_data->torque_des_[0][0], 0, control_fsm_data, MotorType::DMBOT);
        node_1->medulla_cmd_.motor_5 = byte_8.udata;

        //2) node_2 left leg + right arm

        //1. left hip yaw
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[3][0], (float)control_fsm_data->kd_[3][0],
                    (float)control_fsm_data->position_des_[3][0] + (float)control_fsm_data->joint_offset_[3][0], (float)control_fsm_data->velocity_des_[3][0],
                    (float)control_fsm_data->torque_des_[3][0], 14, control_fsm_data, MotorType::INKEXBOT);
        node_2->medulla_cmd_.motor_1 = byte_8.udata;
        //2. left hip roll
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[3][1], (float)control_fsm_data->kd_[3][1],
                    (float)control_fsm_data->position_des_[3][1] + (float)control_fsm_data->joint_offset_[3][1], (float)control_fsm_data->velocity_des_[3][1],
                    (float)control_fsm_data->torque_des_[3][1], 15, control_fsm_data, MotorType::INKEXBOT);
        node_2->medulla_cmd_.motor_2 = byte_8.udata;

        //3. left hip pitch
        temp_tor = pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[3][2], (float)control_fsm_data->kd_[3][2],
                    ((float)control_fsm_data->position_des_[3][2]+ (float)control_fsm_data->joint_offset_[3][2]), (float)control_fsm_data->velocity_des_[3][2],
                    (float)control_fsm_data->torque_des_[3][2], 16, control_fsm_data, MotorType::INKEXBOT);
        node_2->medulla_cmd_.motor_3 = byte_8.udata;
        //node_2->medulla_cmd_.lz_3 = temp_tor & 0xFFFF;

        //4. left knee
        temp_tor = pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[3][3], (float)control_fsm_data->kd_[3][3],
                    ((float)control_fsm_data->position_des_[3][3]+ (float)control_fsm_data->joint_offset_[3][3]), (float)control_fsm_data->velocity_des_[3][3],
                    (float)control_fsm_data->torque_des_[3][3], 17, control_fsm_data, MotorType::INKEXBOT);
        node_2->medulla_cmd_.motor_4 = byte_8.udata;
        //node_2->medulla_cmd_.lz_4 = temp_tor & 0xFFFF;

        //5. left toe
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[3][4], (float)control_fsm_data->kd_[3][4],
                    ((float)control_fsm_data->position_des_[3][4]+ (float)control_fsm_data->joint_offset_[3][4]), (float)control_fsm_data->velocity_des_[3][4],
                    (float)control_fsm_data->torque_des_[3][4], 18, control_fsm_data, MotorType::INKEXBOT);
        node_2->medulla_cmd_.motor_5 = byte_8.udata;

        //6. left toe
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[3][5], (float)control_fsm_data->kd_[3][5],
                    ((float)control_fsm_data->position_des_[3][5]+ (float)control_fsm_data->joint_offset_[3][5]), (float)control_fsm_data->velocity_des_[3][5],
                    (float)control_fsm_data->torque_des_[3][5], 19, control_fsm_data, MotorType::INKEXBOT);
        node_2->medulla_cmd_.motor_6 = byte_8.udata;

        //7. right shoulder pitch
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[0][1], (float)control_fsm_data->kd_[0][1],
                    ((float)control_fsm_data->position_des_[0][1]+ (float)control_fsm_data->joint_offset_[0][1]), (float)control_fsm_data->velocity_des_[0][1],
                    (float)control_fsm_data->torque_des_[0][1], 1, control_fsm_data, MotorType::DMBOT);
        node_2->medulla_cmd_.motor_7 = byte_8.udata;

        //8. right shoulder roll
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[0][2], (float)control_fsm_data->kd_[0][2],
                    ((float)control_fsm_data->position_des_[0][2]+ (float)control_fsm_data->joint_offset_[0][2]), (float)control_fsm_data->velocity_des_[0][2],
                    (float)control_fsm_data->torque_des_[0][2], 2, control_fsm_data, MotorType::DMBOT);
        node_2->medulla_cmd_.motor_8 = byte_8.udata;

        //9. right shoulder yaw
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[0][3], (float)control_fsm_data->kd_[0][3],
                    ((float)control_fsm_data->position_des_[0][3]+ (float)control_fsm_data->joint_offset_[0][3]), (float)control_fsm_data->velocity_des_[0][3],
                    (float)control_fsm_data->torque_des_[0][3], 3, control_fsm_data, MotorType::DMBOT);
        node_2->medulla_cmd_.motor_9 = byte_8.udata;

        //3) node_3 right leg

        //1. right hip yaw
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[2][0], (float)control_fsm_data->kd_[2][0],
                    (float)control_fsm_data->position_des_[2][0] + (float)control_fsm_data->joint_offset_[2][0], (float)control_fsm_data->velocity_des_[2][0],
                    (float)control_fsm_data->torque_des_[2][0], 8, control_fsm_data, MotorType::INKEXBOT);
        node_3->medulla_cmd_.motor_1 = byte_8.udata;

        //2. right hip roll
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[2][1], (float)control_fsm_data->kd_[2][1],
                    (float)control_fsm_data->position_des_[2][1] + (float)control_fsm_data->joint_offset_[2][1], (float)control_fsm_data->velocity_des_[2][1],
                    (float)control_fsm_data->torque_des_[2][1], 9, control_fsm_data, MotorType::INKEXBOT);
        node_3->medulla_cmd_.motor_2 = byte_8.udata;

        //3. right hip pitch
        temp_tor = pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[2][2], (float)control_fsm_data->kd_[2][2],
                    (float)control_fsm_data->position_des_[2][2] + (float)control_fsm_data->joint_offset_[2][2], (float)control_fsm_data->velocity_des_[2][2],
                    (float)control_fsm_data->torque_des_[2][2], 10, control_fsm_data, MotorType::INKEXBOT);
        node_3->medulla_cmd_.motor_3 = byte_8.udata;
        //node_3->medulla_cmd_.lz_3 = temp_tor & 0xFFFF;

        //4. right knee
        temp_tor = pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[2][3], (float)control_fsm_data->kd_[2][3],
                    (float)control_fsm_data->position_des_[2][3] + (float)control_fsm_data->joint_offset_[2][3], (float)control_fsm_data->velocity_des_[2][3],
                    (float)control_fsm_data->torque_des_[2][3], 11, control_fsm_data, MotorType::INKEXBOT);
        node_3->medulla_cmd_.motor_4 = byte_8.udata;
        //node_3->medulla_cmd_.lz_4 = temp_tor & 0xFFFF;

        //4. right toe
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[2][4], (float)control_fsm_data->kd_[2][4],
                    (float)control_fsm_data->position_des_[2][4] + (float)control_fsm_data->joint_offset_[2][4], (float)control_fsm_data->velocity_des_[2][4],
                    (float)control_fsm_data->torque_des_[2][4], 12, control_fsm_data, MotorType::INKEXBOT);
        node_3->medulla_cmd_.motor_5 = byte_8.udata;

        //4. right toe
        pack_pvt_cmd_ex(byte_8.buffer, (float)control_fsm_data->kp_[2][5], (float)control_fsm_data->kd_[2][5],
                    (float)control_fsm_data->position_des_[2][5] + (float)control_fsm_data->joint_offset_[2][5], (float)control_fsm_data->velocity_des_[2][5],
                    (float)control_fsm_data->torque_des_[2][5], 13, control_fsm_data, MotorType::INKEXBOT);
        node_3->medulla_cmd_.motor_6 = byte_8.udata;   
    // //1. left leg
    // // 1. hip yaw
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[3][0], (float)control_fsm_data->kd_[3][0],
    //              (float)control_fsm_data->position_des_[3][0] + (float)control_fsm_data->joint_offset_[3][0], (float)control_fsm_data->velocity_des_[3][0],
    //              (float)control_fsm_data->torque_des_[3][0], 13, control_fsm_data);
    // node_1->medulla_cmd_.motor_1 = byte_8.udata;
                 
    // // 2. hip roll
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[3][1], (float)control_fsm_data->kd_[3][1],
    //              (float)control_fsm_data->position_des_[3][1] + (float)control_fsm_data->joint_offset_[3][1], (float)control_fsm_data->velocity_des_[3][1],
    //              (float)control_fsm_data->torque_des_[3][1], 14, control_fsm_data);
    // node_1->medulla_cmd_.motor_2 = byte_8.udata;
    // //3. hip pitch
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[3][2], (float)control_fsm_data->kd_[3][2],
    //              ((float)control_fsm_data->position_des_[3][2] + (float)control_fsm_data->joint_offset_[3][2]), (float)control_fsm_data->velocity_des_[3][2],
    //              (float)control_fsm_data->torque_des_[3][2], 15, control_fsm_data);
    // node_1->medulla_cmd_.motor_4 = byte_8.udata;
    // //4. knee
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[3][3], (float)control_fsm_data->kd_[3][3],
    //              ((float)control_fsm_data->position_des_[3][3]+ (float)control_fsm_data->joint_offset_[3][3]), (float)control_fsm_data->velocity_des_[3][3],
    //              (float)control_fsm_data->torque_des_[3][3], 16, control_fsm_data);
    // node_1->medulla_cmd_.motor_5 = byte_8.udata;
    // //5. toe
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[3][4], (float)control_fsm_data->kd_[3][4],
    //              ((float)control_fsm_data->position_des_[3][4]+ (float)control_fsm_data->joint_offset_[3][4]), (float)control_fsm_data->velocity_des_[3][4],
    //              (float)control_fsm_data->torque_des_[3][4], 17, control_fsm_data);
    // node_1->medulla_cmd_.motor_6 = byte_8.udata;

    // //2. right leg

    // // 1. hip yaw
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[2][0], (float)control_fsm_data->kd_[2][0],
    //              (float)control_fsm_data->position_des_[2][0] + (float)control_fsm_data->joint_offset_[2][0], (float)control_fsm_data->velocity_des_[2][0],
    //              (float)control_fsm_data->torque_des_[2][0], 8, control_fsm_data);
    // node_2->medulla_cmd_.motor_1 = byte_8.udata;
    // // 2. hip roll
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[2][1], (float)control_fsm_data->kd_[2][1],
    //              (float)control_fsm_data->position_des_[2][1] + (float)control_fsm_data->joint_offset_[2][1], (float)control_fsm_data->velocity_des_[2][1],
    //              (float)control_fsm_data->torque_des_[2][1], 9, control_fsm_data);
    // node_2->medulla_cmd_.motor_2 = byte_8.udata;
    // //3. hip pitch
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[2][2], (float)control_fsm_data->kd_[2][2],
    //              ((float)control_fsm_data->position_des_[2][2]+ (float)control_fsm_data->joint_offset_[2][2]), (float)control_fsm_data->velocity_des_[2][2],
    //              (float)control_fsm_data->torque_des_[2][2], 10, control_fsm_data);
    // node_2->medulla_cmd_.motor_4 = byte_8.udata;
    // //4. knee
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[2][3], (float)control_fsm_data->kd_[2][3],
    //              ((float)control_fsm_data->position_des_[2][3]+ (float)control_fsm_data->joint_offset_[2][3]), (float)control_fsm_data->velocity_des_[2][3],
    //              (float)control_fsm_data->torque_des_[2][3], 11, control_fsm_data);
    // node_2->medulla_cmd_.motor_5 = byte_8.udata;
    // //5. toe
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[2][4], (float)control_fsm_data->kd_[2][4],
    //              ((float)control_fsm_data->position_des_[2][4]+ (float)control_fsm_data->joint_offset_[2][4]), (float)control_fsm_data->velocity_des_[2][4],
    //              (float)control_fsm_data->torque_des_[2][4], 12, control_fsm_data);
    // node_2->medulla_cmd_.motor_6 = byte_8.udata;
    // // 3. right arm
    // //1. shoulder pitch
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[0][0], (float)control_fsm_data->kd_[0][0],
    //              (float)control_fsm_data->position_des_[0][0] + (float)control_fsm_data->joint_offset_[0][0], (float)control_fsm_data->velocity_des_[0][0],
    //              (float)control_fsm_data->torque_des_[0][0], 0, control_fsm_data);
    // node_3->medulla_cmd_.motor_1 = byte_8.udata;           

    // //2. shoulder roll
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[0][1], (float)control_fsm_data->kd_[0][1],
    //              (float)control_fsm_data->position_des_[0][1] + (float)control_fsm_data->joint_offset_[0][1], (float)control_fsm_data->velocity_des_[0][1],
    //              (float)control_fsm_data->torque_des_[0][1], 1, control_fsm_data);
    // node_3->medulla_cmd_.motor_2 = byte_8.udata;    
    // //3. shoulder yaw
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[0][2], (float)control_fsm_data->kd_[0][2],
    //              (float)control_fsm_data->position_des_[0][2] + (float)control_fsm_data->joint_offset_[0][2], (float)control_fsm_data->velocity_des_[0][2],
    //              (float)control_fsm_data->torque_des_[0][2], 2, control_fsm_data);
    // node_3->medulla_cmd_.motor_3 = byte_8.udata;    
    // //4. elbow pitch
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[0][3], (float)control_fsm_data->kd_[0][3],
    //              (float)control_fsm_data->position_des_[0][3] + (float)control_fsm_data->joint_offset_[0][3], (float)control_fsm_data->velocity_des_[0][3],
    //              (float)control_fsm_data->torque_des_[0][3], 3, control_fsm_data);
    // node_3->medulla_cmd_.motor_4 = byte_8.udata;    
    // // 4. left arm
    // //1. shoulder pitch
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[1][0], (float)control_fsm_data->kd_[1][0],
    //              (float)control_fsm_data->position_des_[1][0] + (float)control_fsm_data->joint_offset_[1][0], (float)control_fsm_data->velocity_des_[1][0],
    //              (float)control_fsm_data->torque_des_[1][0], 4, control_fsm_data);
    // node_4->medulla_cmd_.motor_1 = byte_8.udata;
    // //2. shoulder roll
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[1][1], (float)control_fsm_data->kd_[1][1],
    //              (float)control_fsm_data->position_des_[1][1] + (float)control_fsm_data->joint_offset_[1][1], (float)control_fsm_data->velocity_des_[1][1],
    //              (float)control_fsm_data->torque_des_[1][1], 5, control_fsm_data);
    // node_4->medulla_cmd_.motor_2 = byte_8.udata;
    // //3. shoulder yaw
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[1][2], (float)control_fsm_data->kd_[1][2],
    //              (float)control_fsm_data->position_des_[1][2] + (float)control_fsm_data->joint_offset_[1][2], (float)control_fsm_data->velocity_des_[1][2],
    //              (float)control_fsm_data->torque_des_[1][2], 6, control_fsm_data);
    // node_4->medulla_cmd_.motor_3 = byte_8.udata;
    // //4. elbow pitch
    // pack_pvt_cmd(byte_8.buffer, (float)control_fsm_data->kp_[1][3], (float)control_fsm_data->kd_[1][3],
    //              (float)control_fsm_data->position_des_[1][3] + (float)control_fsm_data->joint_offset_[1][3], (float)control_fsm_data->velocity_des_[1][3],
    //              (float)control_fsm_data->torque_des_[1][3], 7, control_fsm_data);
    // node_4->medulla_cmd_.motor_4 = byte_8.udata;
    /*
      第三步 命令下发
    */
    // cmd write
    rt_ethercat_set_command();
    // 这个函数看起来是用于实时以太网通信的关键部分，它负责发送和接收数据，并对通信质量进行监控和错误处理。
    rt_ethercat_run();
  }


  bool dora2HW::setupJoints()
  {
    for (const auto &joint : urdfModel_->joints_)
    {
      int leg_index=0;
      int joint_index=0; 
      int index=0;
      if (joint.first.find("leg_l") != std::string::npos)
      {
        leg_index = 0;
        index+=leg_index*6;
        // leg_index = UNITREE_LEGGED_SDK::FR_;
      }
      else if (joint.first.find("leg_r") != std::string::npos)
      {
        leg_index = 1;
        index+=leg_index*6;
        // leg_index = UNITREE_LEGGED_SDK::RR_;
      }

      else
        continue;  // 不是左腿或右腿的关节，跳过

      // 根据关节名称确定关节在腿上的索引
      if (joint.first.find("1_joint") != std::string::npos)
        joint_index = 0;
      else if (joint.first.find("2_joint") != std::string::npos)
        joint_index = 1;
      else if (joint.first.find("3_joint") != std::string::npos)
        joint_index = 2;
      else if (joint.first.find("4_joint") != std::string::npos)
        joint_index = 3;
      else if (joint.first.find("5_joint") != std::string::npos)
        joint_index = 4;
      else if (joint.first.find("6_joint") != std::string::npos)
        joint_index = 5;
      else
        continue;  // 不是1-6号关节的关节，跳过

      // 计算该关节在joint_data_数组中的索引
      index+=joint_index;
      ROS_INFO("joint index = %d", index);

      // 创建JointStateHandle对象并注册到jointStateInterface_
      hardware_interface::JointStateHandle state_handle(joint.first, &joint_data_[index].pos_, &joint_data_[index].vel_,
                                                        &joint_data_[index].tau_);
      jointStateInterface_.registerHandle(state_handle);

      // 创建HybridJointHandle对象并注册到hybridJointInterface_
      hybridJointInterface_.registerHandle(HybridJointHandle(state_handle, &joint_data_[index].pos_des_,
                                                            &joint_data_[index].vel_des_, &joint_data_[index].kp_,
                                                            &joint_data_[index].kd_, &joint_data_[index].ff_));
    } 
    return true;
  }


  bool dora2HW::setupImu()
  {
     imuSensorInterface_.registerHandle(hardware_interface::ImuSensorHandle(
        "base_imu", "base_imu", imu_data_.ori, imu_data_.ori_cov, imu_data_.angular_vel, imu_data_.angular_vel_cov,
        imu_data_.linear_acc, imu_data_.linear_acc_cov));
    imu_data_.ori_cov[0] = 0.0012;
    imu_data_.ori_cov[4] = 0.0012;
    imu_data_.ori_cov[8] = 0.0012;

    imu_data_.angular_vel_cov[0] = 0.0004;
    imu_data_.angular_vel_cov[4] = 0.0004;
    imu_data_.angular_vel_cov[8] = 0.0004;

    return true;
  }

  void dora2HW::motor_setting(const std::shared_ptr<ControlFSMData>& a)
  {
    // right arm(0:3) -> left arm(4:7) -> right leg(8:12) -> left leg(13:17)
    if(a == nullptr){
      std::cout<<"a == nullptr"<<__LINE__<<std::endl;
      return;
    }
    
    try {
        std::cout<<" a->joint_max_current_[0]"<< a->joint_max_current_[0]<<__LINE__<<std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
    }
    try {
        std::cout<<"config.motor_max_current[config.right_arm_motor_type[0]]"<<config.motor_max_current[config.right_arm_motor_type[0]]<<__LINE__<<std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
    }
    a->joint_max_current_[0] = config.motor_max_current[config.right_arm_motor_type[0]]; 
    a->joint_max_current_[1] = config.motor_max_current[config.right_arm_motor_type[1]];
    a->joint_max_current_[2] = config.motor_max_current[config.right_arm_motor_type[2]];
    a->joint_max_current_[3] = config.motor_max_current[config.right_arm_motor_type[3]];

    a->joint_max_current_[4] = config.motor_max_current[config.left_arm_motor_type[0]];
    a->joint_max_current_[5] = config.motor_max_current[config.left_arm_motor_type[1]];
    a->joint_max_current_[6] = config.motor_max_current[config.left_arm_motor_type[2]];
    a->joint_max_current_[7] = config.motor_max_current[config.left_arm_motor_type[3]];

    a->joint_max_current_[8] = config.motor_max_current[config.right_leg_motor_type[0]];
    a->joint_max_current_[9] = config.motor_max_current[config.right_leg_motor_type[1]];
    a->joint_max_current_[10] = config.motor_max_current[config.right_leg_motor_type[2]];
    a->joint_max_current_[11] = config.motor_max_current[config.right_leg_motor_type[3]];
    a->joint_max_current_[12] = config.motor_max_current[config.right_leg_motor_type[4]];

    a->joint_max_current_[13] = config.motor_max_current[config.left_leg_motor_type[0]];
    a->joint_max_current_[14] = config.motor_max_current[config.left_leg_motor_type[1]];
    a->joint_max_current_[15] = config.motor_max_current[config.left_leg_motor_type[2]];
    a->joint_max_current_[16] = config.motor_max_current[config.left_leg_motor_type[3]];
    a->joint_max_current_[17] = config.motor_max_current[config.left_leg_motor_type[4]];

    a->joint_min_current_[0] = config.motor_min_current[config.right_arm_motor_type[0]];
    a->joint_min_current_[1] = config.motor_min_current[config.right_arm_motor_type[1]];
    a->joint_min_current_[2] = config.motor_min_current[config.right_arm_motor_type[2]];
    a->joint_min_current_[3] = config.motor_min_current[config.right_arm_motor_type[3]];

    a->joint_min_current_[4] = config.motor_min_current[config.left_arm_motor_type[0]];
    a->joint_min_current_[5] = config.motor_min_current[config.left_arm_motor_type[1]];
    a->joint_min_current_[6] = config.motor_min_current[config.left_arm_motor_type[2]];
    a->joint_min_current_[7] = config.motor_min_current[config.left_arm_motor_type[3]];

    a->joint_min_current_[8] = config.motor_min_current[config.right_leg_motor_type[0]];
    a->joint_min_current_[9] = config.motor_min_current[config.right_leg_motor_type[1]];
    a->joint_min_current_[10] = config.motor_min_current[config.right_leg_motor_type[2]];
    a->joint_min_current_[11] = config.motor_min_current[config.right_leg_motor_type[3]];
    a->joint_min_current_[12] = config.motor_min_current[config.right_leg_motor_type[4]];

    a->joint_min_current_[13] = config.motor_min_current[config.left_leg_motor_type[0]];
    a->joint_min_current_[14] = config.motor_min_current[config.left_leg_motor_type[1]];
    a->joint_min_current_[15] = config.motor_min_current[config.left_leg_motor_type[2]];
    a->joint_min_current_[16] = config.motor_min_current[config.left_leg_motor_type[3]];
    a->joint_min_current_[17] = config.motor_min_current[config.left_leg_motor_type[4]];


    a->joint_max_torque_[0] = config.motor_max_torque[config.right_arm_motor_type[0]];
    a->joint_max_torque_[1] = config.motor_max_torque[config.right_arm_motor_type[1]];
    a->joint_max_torque_[2] = config.motor_max_torque[config.right_arm_motor_type[2]];
    a->joint_max_torque_[3] = config.motor_max_torque[config.right_arm_motor_type[3]];

    a->joint_max_torque_[4] = config.motor_max_torque[config.left_arm_motor_type[0]];
    a->joint_max_torque_[5] = config.motor_max_torque[config.left_arm_motor_type[1]];
    a->joint_max_torque_[6] = config.motor_max_torque[config.left_arm_motor_type[2]];
    a->joint_max_torque_[7] = config.motor_max_torque[config.left_arm_motor_type[3]];

    a->joint_max_torque_[8] = config.motor_max_torque[config.right_leg_motor_type[0]];
    a->joint_max_torque_[9] = config.motor_max_torque[config.right_leg_motor_type[1]];
    a->joint_max_torque_[10] = config.motor_max_torque[config.right_leg_motor_type[2]];
    a->joint_max_torque_[11] = config.motor_max_torque[config.right_leg_motor_type[3]];
    a->joint_max_torque_[12] = config.motor_max_torque[config.right_leg_motor_type[4]];

    a->joint_max_torque_[13] = config.motor_max_torque[config.left_leg_motor_type[0]];
    a->joint_max_torque_[14] = config.motor_max_torque[config.left_leg_motor_type[1]];
    a->joint_max_torque_[15] = config.motor_max_torque[config.left_leg_motor_type[2]];
    a->joint_max_torque_[16] = config.motor_max_torque[config.left_leg_motor_type[3]];
    a->joint_max_torque_[17] = config.motor_max_torque[config.left_leg_motor_type[4]];

    a->joint_min_torque_[0] = config.motor_min_torque[config.right_arm_motor_type[0]];
    a->joint_min_torque_[1] = config.motor_min_torque[config.right_arm_motor_type[1]];
    a->joint_min_torque_[2] = config.motor_min_torque[config.right_arm_motor_type[2]];
    a->joint_min_torque_[3] = config.motor_min_torque[config.right_arm_motor_type[3]];

    a->joint_min_torque_[4] = config.motor_min_torque[config.left_arm_motor_type[0]];
    a->joint_min_torque_[5] = config.motor_min_torque[config.left_arm_motor_type[1]];
    a->joint_min_torque_[6] = config.motor_min_torque[config.left_arm_motor_type[2]];
    a->joint_min_torque_[7] = config.motor_min_torque[config.left_arm_motor_type[3]];

    a->joint_min_torque_[8] = config.motor_min_torque[config.right_leg_motor_type[0]];
    a->joint_min_torque_[9] = config.motor_min_torque[config.right_leg_motor_type[1]];
    a->joint_min_torque_[10] = config.motor_min_torque[config.right_leg_motor_type[2]];
    a->joint_min_torque_[11] = config.motor_min_torque[config.right_leg_motor_type[3]];
    a->joint_min_torque_[12] = config.motor_min_torque[config.right_leg_motor_type[4]];

    a->joint_min_torque_[13] = config.motor_min_torque[config.left_leg_motor_type[0]];
    a->joint_min_torque_[14] = config.motor_min_torque[config.left_leg_motor_type[1]];
    a->joint_min_torque_[15] = config.motor_min_torque[config.left_leg_motor_type[2]];
    a->joint_min_torque_[16] = config.motor_min_torque[config.left_leg_motor_type[3]];
    a->joint_min_torque_[17] = config.motor_min_torque[config.left_leg_motor_type[4]];

    //for (int i = 0; i < 5; ++i) {
        //a->left_leg_direction_[i] = config.left_leg_motor_dir[i];
        //a->right_leg_direction_[i] = config.right_leg_motor_dir[i];
   // }

    // right arm(0:3) -> left arm(4:7) -> right leg(8:12) -> left leg(13:17)
    a->joint_dir_[0] = config.right_leg_motor_dir[0];
    a->joint_dir_[1] = config.right_leg_motor_dir[1];
    a->joint_dir_[2] = config.right_leg_motor_dir[2];
    a->joint_dir_[3] = config.right_leg_motor_dir[3];

    a->joint_dir_[4] = config.left_leg_motor_dir[0];
    a->joint_dir_[5] = config.left_leg_motor_dir[1];
    a->joint_dir_[6] = config.left_leg_motor_dir[2];
    a->joint_dir_[7] = config.left_leg_motor_dir[3];



    a->joint_dir_[8] = config.right_leg_motor_dir[0];
    a->joint_dir_[9] = config.right_leg_motor_dir[1];
    a->joint_dir_[10] = config.right_leg_motor_dir[2];
    a->joint_dir_[11] = config.right_leg_motor_dir[3];
    a->joint_dir_[12] = config.right_leg_motor_dir[4];

    a->joint_dir_[13] = config.left_leg_motor_dir[0];
    a->joint_dir_[14] = config.left_leg_motor_dir[1];
    a->joint_dir_[15] = config.left_leg_motor_dir[2];
    a->joint_dir_[16] = config.left_leg_motor_dir[3];
    a->joint_dir_[17] = config.left_leg_motor_dir[4];


    //TODO: config
    for(int i = 0; i < 4; ++i) {
        for (int j = 0; j < 6; ++j) {
            a->joint_offset_[i][j] = 0;
        }
    }
  }
  void dora2HW::CalibrateImu(const bool is_sim) {
   // std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
   // std::cout << "[quat] : " << robot_state->quat << std::endl;
   // std::cout << "[euler]: " << robot_state->imu_rpy_angle.transpose() << std::endl;
   // std::cout << "[gyro] : " << robot_state->imu_rpy_rate.transpose() << std::endl;
   // std::cout << "[acc]  : " << robot_state->imu_xyz_acc.transpose() << std::endl;

    // get ImuCalibrationParam
    Eigen::Quaternion<double> quat_pre;
    Eigen::Quaternion<double> quat_post;
    if (is_sim) {
        quat_pre.x() = 0;
        quat_pre.y() = 0;
        quat_pre.z() = 0;
        quat_pre.w() = 1;
        quat_post.x() = 0;
        quat_post.y() = 0;
        quat_post.z() = 0;
        quat_post.w() = 1;
    } else {
        quat_pre.x() = 1;
        quat_pre.y() = 0;
        quat_pre.z() = 0;
        quat_pre.w() = 0;
        quat_post.x() = 0;
        quat_post.y() = 1;
        quat_post.z() = 0;
        quat_post.w() = 0;
    }
    Eigen::Matrix3d mat_pre;
    Eigen::Matrix3d mat_post;
    Eigen::Matrix3d mat_output;
    mat_pre = ori_tools_.quatToRotMatrix(quat_pre);
    mat_post = ori_tools_.quatToRotMatrix(quat_post);

    // solve quat AKA orientation
    Eigen::Quaternion<double> quat_input;
    Eigen::Quaternion<double> quat_output;

    quat_input.x() = (double)imu_rc->imu_rc_data_.q1;
    quat_input.y() = (double)imu_rc->imu_rc_data_.q2;
    quat_input.z() = (double)imu_rc->imu_rc_data_.q3;
    quat_input.w() = (double)imu_rc->imu_rc_data_.q0;

    mat_output = mat_pre * ori_tools_.quatToRotMatrix(quat_input) * mat_post;

    quat_output = ori_tools_.rotationMatrixToQuaternion(mat_output);

    imu_data_.ori[0] = quat_output.x();
    imu_data_.ori[1] = quat_output.y();
    imu_data_.ori[2] = quat_output.z();
    imu_data_.ori[3] = quat_output.w();

    // rpy angle
    // robot_state->imu_rpy_angle = ori_tools_.quatToEulerAngle(quat_output);

    // solve gyro AKA angular velocity
    Eigen::Vector3d gyro_input;
    Eigen::Vector3d gyro_output;

    gyro_input << (double)imu_rc->imu_rc_data_.gyr_x,
                  (double)imu_rc->imu_rc_data_.gyr_y,
                  (double)imu_rc->imu_rc_data_.gyr_z;
      
    gyro_output = mat_post * gyro_input;

    imu_data_.angular_vel[0] = gyro_output(0);
    imu_data_.angular_vel[1] = gyro_output(1);
    imu_data_.angular_vel[2] = gyro_output(2);

    // solve acc AKA linear acceleration
    Eigen::Vector3d acc_input;
    Eigen::Vector3d acc_output;

    acc_input << (double)imu_rc->imu_rc_data_.acc_x,
                 (double)imu_rc->imu_rc_data_.acc_y,
                 (double)imu_rc->imu_rc_data_.acc_z;
    if(is_sim) {
        acc_output = mat_post * acc_input;
    } else {
        acc_output = mat_post * acc_input * 9.81;
    }

    imu_data_.linear_acc[0] = acc_output(0);
    imu_data_.linear_acc[1] = acc_output(1);
    imu_data_.linear_acc[2] = acc_output(2);
  }

} // namespace legged
