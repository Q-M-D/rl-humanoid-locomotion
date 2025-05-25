#pragma once

#include "frame.h"
#include <fstream>

void add_data(std::ofstream& outfile, const VectorXd& data, const int& data_num);

bool record_debug_data(const std::shared_ptr<Frame>& frame_ptr, 
    const std::string& path);


