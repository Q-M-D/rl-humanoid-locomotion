add_definitions(-std=c++17)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CXX_COMPILER g++-9)

include_directories(".")
include_directories("../third-party/ParamHandler")
include_directories("../third-party/inih")
include_directories("../third-party/SOEM/osal")
include_directories("../third-party/SOEM/osal/linux")
include_directories("../third-party/SOEM/oshw")
include_directories("../third-party/SOEM/oshw/linux")
include_directories("../third-party/SOEM/soem")

# 递归地收集所有 .cpp 文件
file(GLOB_RECURSE sources "*.cpp")

# 查找并配置 Threads 库
find_package(Threads)

# 查找并配置 glog 库，并设置为必需
#find_package(glog REQUIRED)

# 创建一个名为 core 的共享库，并使用 sources 变量中的所有 .cpp 文件
add_library(core SHARED ${sources})

# 链接库
#target_link_libraries(core inih dynacore_param_handler soem Threads::Threads glog::glog)
target_link_libraries(core inih dynacore_param_handler soem Threads::Threads)

