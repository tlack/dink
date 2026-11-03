cmake_minimum_required(VERSION 3.16)
# MMRT from the BabyTalk submodule, and Dink's engine.
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../babytalk/components/mmrt"
                         "${CMAKE_CURRENT_LIST_DIR}/../components/dink")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(dink_test)
