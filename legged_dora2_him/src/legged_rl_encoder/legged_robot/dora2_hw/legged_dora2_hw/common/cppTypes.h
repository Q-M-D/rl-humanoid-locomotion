/*! @file cTypes.h
 *  @brief Common types that are only valid in C++
 *
 *  This file contains types which are only used in C++ code.  This includes
 * Eigen types, template types, aliases, ...
 */

#ifndef PROJECT_CPPTYPES_H
#define PROJECT_CPPTYPES_H

#include <vector>
#include <algorithm>
#include <memory>
#include "cTypes.h"
#include "log.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <limits>
#include <eigen3/Eigen/StdVector>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Matrix3f;
using Eigen::Vector3f;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::AngleAxisd;
using Eigen::Quaterniond;

enum class CoordinateAxis {X = 1, Y, Z, Unknown};
enum class TestPhase {WAITING = 1, 
                    ROATE_TO_NONZERO_POS, 
                    ROATE_TO_ZERO_POS,
                    EXEC_TORQUE_CMD,
                    UNKNOWN};

template <typename Type, int Size>
using Vector = Matrix<Type, Size, 1>;

using Matrix4d = Matrix<double, 4, 4>;

// Rotation Matrix
template <typename T>
using RotMat = typename Eigen::Matrix<T, 3, 3>;

// 2x1 Vector
template <typename T>
using Vec2 = typename Eigen::Matrix<T, 2, 1>;

// 3x1 Vector
template <typename T>
using Vec3 = typename Eigen::Matrix<T, 3, 1>;

template <typename T>
using Vec4 = typename Eigen::Matrix<T, 4, 1>;

template <typename T>
using Vec5 = typename Eigen::Matrix<T, 5, 1>;

// 3x3 Matrix
template <typename T>
using Mat3 = typename Eigen::Matrix<T, 3, 3>;

// // 4x1 Vector
template <typename T>
using Quat = typename Eigen::Matrix<T, 4, 1>;

// // Spatial Vector (6x1, all subspaces)
template <typename T>
using SVec = typename Eigen::Matrix<T, 6, 1>;

// // Spatial Transform (6x6)
// template <typename T>
// using SXform = typename Eigen::Matrix<T, 6, 6>;

// // 6x6 Matrix
template <typename T>
using Mat6 = typename Eigen::Matrix<T, 6, 6>;

// // Dynamically sized vector
template <typename T>
using DVec = typename Eigen::Matrix<T, Eigen::Dynamic, 1>;

// Dynamically sized matrix
template <typename T>
using DMat = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

// Dynamically sized matrix with cartesian vector columns
template <typename T>
using D3Mat = typename Eigen::Matrix<T, 3, Eigen::Dynamic>;

union Byte8
{
    uint64_t udata;
    uint8_t buffer[8];
};

union Byte2
{
    int16_t data;
    uint16_t udata;
    uint8_t buffer[2];
};

union Byte4 {
    uint32_t udata;
    int32_t sdata;
    float fdata;
    uint8_t data[4];
};

struct RobotModel {
    int robot_type = 1;

    int single_arm_freedom = 4;
    int single_leg_freedom = 6;
    int total_joint_freedom = 20;
};

struct RobotState {
    RobotState(){}
    RobotState(const int single_arm_freedom, const int single_leg_freedom) {
        left_arm_joint_pos = VectorXd::Zero(single_arm_freedom);
        right_arm_joint_pos = VectorXd::Zero(single_arm_freedom);
        left_arm_joint_vel = VectorXd::Zero(single_arm_freedom);
        right_arm_joint_vel = VectorXd::Zero(single_arm_freedom);
        left_arm_joint_tor = VectorXd::Zero(single_arm_freedom);
        right_arm_joint_tor = VectorXd::Zero(single_arm_freedom);
	    left_arm_joint_error = VectorXi::Zero(single_arm_freedom);
        right_arm_joint_error = VectorXi::Zero(single_arm_freedom);

        left_leg_joint_pos = VectorXd::Zero(single_leg_freedom);
        right_leg_joint_pos = VectorXd::Zero(single_leg_freedom);
        left_leg_joint_vel = VectorXd::Zero(single_leg_freedom);
        right_leg_joint_vel = VectorXd::Zero(single_leg_freedom);
        left_leg_joint_tor = VectorXd::Zero(single_leg_freedom);
        right_leg_joint_tor = VectorXd::Zero(single_leg_freedom);
        left_leg_joint_error = VectorXi::Zero(single_leg_freedom);
        right_leg_joint_error = VectorXi::Zero(single_leg_freedom);
    }
    Vector<int, Dynamic> left_arm_joint_error, right_arm_joint_error,
        left_leg_joint_error, right_leg_joint_error;
    Vector<double, Dynamic> left_arm_joint_pos, right_arm_joint_pos,
        left_leg_joint_pos, right_leg_joint_pos;
    Vector<double, Dynamic> left_arm_joint_vel, right_arm_joint_vel,
        left_leg_joint_vel, right_leg_joint_vel;
    Vector<double, Dynamic> left_arm_joint_tor, right_arm_joint_tor,
        left_leg_joint_tor, right_leg_joint_tor;

    Vector3d imu_rpy_angle = Vector3d::Zero(); 
    Vector3d imu_rpy_rate = Vector3d::Zero();
    Vector3d imu_xyz_acc = Vector3d::Zero();

    TestPhase test_phase = TestPhase::WAITING;
};

struct WalkPhaseTransitionData {
    WalkPhaseTransitionData(const int single_arm_freedom, const int single_leg_freedom) {
        last_phase_end_robot_state = RobotState(single_arm_freedom, single_leg_freedom);
    }
    double last_phase_end_time = 0;
    TestPhase test_phase = TestPhase::UNKNOWN;
    RobotState last_phase_end_robot_state;
};
struct UserCommand {
//update keyboard's output
    char keyboard_button;
};
struct RobotCommand {
    bool is_rotate_to_nonzero_pos = false;
    bool is_rotate_to_zero_pos = false;
    bool is_only_exec_torque_cmd = false;
    bool is_stop = false;
    bool is_print_joint_debug = false;
    bool is_calibrate_cur_pos_as_zero = false;
    bool is_sin_wave = false;
    bool is_swing_cmd = false;
    bool is_torque_cmd = false;
};
struct Command {
    UserCommand user_cmd;
    RobotCommand robot_cmd;
};

struct JointCommand {
    JointCommand(const int single_arm_freedom, const int single_leg_freedom) {
        left_arm_joint_torcmd = VectorXd::Zero(single_arm_freedom);
        right_arm_joint_torcmd = VectorXd::Zero(single_arm_freedom);
        left_leg_joint_torcmd = VectorXd::Zero(single_leg_freedom);
        right_leg_joint_torcmd = VectorXd::Zero(single_leg_freedom);

        left_arm_joint_poscmd = VectorXd::Zero(single_arm_freedom);
        right_arm_joint_poscmd = VectorXd::Zero(single_arm_freedom);
        left_leg_joint_poscmd = VectorXd::Zero(single_leg_freedom);
        right_leg_joint_poscmd = VectorXd::Zero(single_leg_freedom);

        left_arm_joint_velcmd = VectorXd::Zero(single_arm_freedom);
        right_arm_joint_velcmd = VectorXd::Zero(single_arm_freedom);
        left_leg_joint_velcmd = VectorXd::Zero(single_leg_freedom);
        right_leg_joint_velcmd = VectorXd::Zero(single_leg_freedom);

        left_arm_joint_kp = VectorXd::Zero(single_arm_freedom);
        right_arm_joint_kp = VectorXd::Zero(single_arm_freedom);
        left_leg_joint_kp = VectorXd::Zero(single_leg_freedom);
        right_leg_joint_kp = VectorXd::Zero(single_leg_freedom);

        left_arm_joint_kd = VectorXd::Zero(single_arm_freedom);
        right_arm_joint_kd = VectorXd::Zero(single_arm_freedom);
        left_leg_joint_kd = VectorXd::Zero(single_leg_freedom);
        right_leg_joint_kd = VectorXd::Zero(single_leg_freedom);
    }

    void zero_joint_cmd() {
        left_arm_joint_torcmd.setZero();
        right_arm_joint_torcmd.setZero();
        left_leg_joint_torcmd.setZero();
        right_leg_joint_torcmd.setZero();

        left_arm_joint_poscmd.setZero();
        right_arm_joint_poscmd.setZero();
        left_leg_joint_poscmd.setZero();
        right_leg_joint_poscmd.setZero();

        left_arm_joint_velcmd.setZero();
        right_arm_joint_velcmd.setZero();
        left_leg_joint_velcmd.setZero();
        right_leg_joint_velcmd.setZero();

        left_arm_joint_kp.setZero();
        right_arm_joint_kp.setZero();
        left_leg_joint_kp.setZero();
        right_leg_joint_kp.setZero();

        left_arm_joint_kd.setZero();
        right_arm_joint_kd.setZero();
        left_leg_joint_kd.setZero();
        right_leg_joint_kd.setZero();
    }
    void zero_arm_joint_cmd() {
        left_arm_joint_torcmd.setZero();
        right_arm_joint_torcmd.setZero();

        left_arm_joint_poscmd.setZero();
        right_arm_joint_poscmd.setZero();

        left_arm_joint_velcmd.setZero();
        right_arm_joint_velcmd.setZero();

        left_arm_joint_kp.setZero();
        right_arm_joint_kp.setZero();

        left_arm_joint_kd.setZero();
        right_arm_joint_kd.setZero();
    }

    Vector<double, Dynamic> left_arm_joint_torcmd, right_arm_joint_torcmd,
        left_leg_joint_torcmd, right_leg_joint_torcmd;
    Vector<double, Dynamic> left_arm_joint_poscmd, right_arm_joint_poscmd,
        left_leg_joint_poscmd, right_leg_joint_poscmd;
    Vector<double, Dynamic> left_arm_joint_velcmd, right_arm_joint_velcmd,
        left_leg_joint_velcmd, right_leg_joint_velcmd;
    Vector<double, Dynamic> left_arm_joint_kp, right_arm_joint_kp,
        left_leg_joint_kp, right_leg_joint_kp;
    Vector<double, Dynamic> left_arm_joint_kd, right_arm_joint_kd,
        left_leg_joint_kd, right_leg_joint_kd;
};

#endif  // PROJECT_CPPTYPES_H
