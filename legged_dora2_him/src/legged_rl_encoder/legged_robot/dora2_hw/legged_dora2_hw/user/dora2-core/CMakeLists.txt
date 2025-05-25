include_directories("../../common/")
include_directories("./")
include_directories("../../third-party/ParamHandler")
include_directories("../../third-party/SOEM/osal")
include_directories("../../third-party/SOEM/osal/linux")
include_directories("../../third-party/SOEM/oshw")
include_directories("../../third-party/SOEM/oshw/linux")
include_directories("../../third-party/SOEM/soem")
# add
find_package(catkin REQUIRED
        COMPONENTS
        roscpp
        legged_hw
        tf
        )
find_package(Eigen3 REQUIRED)
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
# catkin_package(
#         INCLUDE_DIRS
#         include
#         CATKIN_DEPENDS
#         roscpp
#         legged_hw
# )
find_package(Boost REQUIRED)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${Eigen3_INCLUDE_DIRS} 
)

find_package(Threads)

file(GLOB_RECURSE sources               
        "../../src/*.cpp"
        "FSM_States/*.cpp"
        "hardware/*.cpp"
        "../../../../../legged_base/legged_hw/src/*.cpp"
        )

add_executable(dora2-core  ${sources})
target_link_libraries(dora2-core 
                        core 
                        dynacore_param_handler 
                        Threads::Threads 
                        )