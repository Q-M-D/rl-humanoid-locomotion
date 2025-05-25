cmake_minimum_required(VERSION 3.10)
project(rl_controllers)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Find catkin macros and libraries
find_package(catkin REQUIRED
        COMPONENTS
        roscpp
        legged_common
        dynamic_reconfigure
        controller_interface
        std_msgs
        robot_state_publisher
        message_generation
        rospy
        )

find_package(Eigen3 3.3 REQUIRED NO_MODULE)



catkin_package(
        INCLUDE_DIRS
        include
        ${EIGEN3_INCLUDE_DIRS}
        LIBRARIES
        ${PROJECT_NAME}
        CATKIN_DEPENDS
        roscpp
        legged_common
        controller_interface
        dynamic_reconfigure
        DEPENDS
        onnxruntime
        std_msgs
        message_generation
        robot_state_publisher
        tf
        rospy
)


###########
## Build ##
###########
catkin_install_python(PROGRAMS
  python_part/node1.py 
  python_part/Contact_Listener.py 
  python_part/Joint_Listener.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
)


## Declare a cpp library
add_library(${PROJECT_NAME}
        src/RLControllerBase.cpp
        src/AcController.cpp
        src/utilities.cpp 
        )

        # 寻找pkg-config工具
find_package(PkgConfig REQUIRED)

# 尝试使用pkg-config查找onnxruntime库
pkg_check_modules(ONNXRUNTIME QUIET libonnxruntime)

# 如果找不到onnxruntime库，则手动添加
if(NOT ONNXRUNTIME_FOUND)
        find_package(onnxruntime REQUIRED)
endif()

# 添加链接库
target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        onnxruntime     
        )

#########################
###   CLANG TOOLING   ###
#########################
find_package(cmake_clang_tools QUIET)
if (cmake_clang_tools_FOUND)
    message(STATUS "Run clang tooling for target " ${PROJECT_NAME})
    add_clang_tooling(
            TARGETS ${PROJECT_NAME} legged_target_trajectories_publisher
            SOURCE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
            CT_HEADER_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
            CF_WERROR
    )
endif (cmake_clang_tools_FOUND)

#############
## Install ##
#############


# Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

# Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h"
        )

# Install plugins
install(FILES legged_controllers_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )
