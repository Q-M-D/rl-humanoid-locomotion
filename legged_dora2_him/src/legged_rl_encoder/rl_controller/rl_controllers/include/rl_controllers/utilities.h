#pragma once
#include "rl_controllers/Types.h"

#include "std_msgs/Float64MultiArray.h"

namespace legged {

// using namespace ocs2;

std_msgs::Float64MultiArray createFloat64MultiArrayFromVector(const vector_t& data);

}