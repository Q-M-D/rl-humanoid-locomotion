#pragma once

#include "cppTypes.h"

class Interpolate {
public:
    Interpolate() {}

    void cubic_spline_pos_and_vel(const double init_pos, const double target_pos,
        const double init_vel, const double target_vel,
        const double now_time, const double exec_time, const double interpolate_period,
        double* ref_pos, double* ref_vel) {
        if (ref_pos == nullptr || ref_vel == nullptr) {
            std::cout << "nullptr, skip cal cubic_spline" << std::endl;
            return;
        }
        double cur_time = now_time;
        double a, b, c, d;
        const double dt = interpolate_period;
        double cur_pos, next_pos;
        d = init_pos;
        c = init_vel;
        b = (3 * target_pos - target_vel * exec_time - 2 * init_vel * exec_time - 3 * init_pos) / pow(exec_time, 2);
        a = (target_vel * exec_time - 2 * target_pos + init_vel * exec_time + 2 * init_pos) / pow(exec_time, 3);

        if (cur_time > exec_time)
            cur_time = exec_time;
        cur_pos = a * pow(cur_time, 3) + b * pow(cur_time, 2) + c * cur_time + d;

        if (cur_time + dt > exec_time)
            cur_time = exec_time - dt;
        next_pos = a * pow(cur_time + dt, 3) + b * pow(cur_time + dt, 2) + c * (cur_time + dt) + d;

        *ref_pos = cur_pos;
        *ref_vel = (next_pos - cur_pos) / dt;
    }
    
private:
    const double pi = 3.1415926;
};