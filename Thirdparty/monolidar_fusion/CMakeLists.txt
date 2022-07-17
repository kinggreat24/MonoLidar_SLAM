# Modify only if you know what you are doing!
cmake_minimum_required(VERSION 2.8.12)
project(monolidar_fusion)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


###################
## find packages ##
###################

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)

set(PCL_DIR /usr/local/pcl_1.7/share/pcl-1.7)
find_package(PCL REQUIRED)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

##CERES
set(Ceres_DIR /usr/local/ceres-1.14/lib/cmake/Ceres)
find_package(Ceres REQUIRED)

message(STATUS "Ceres include : ${CERES_INCLUDE_DIRS}")
message(STATUS "CERES_LIBRARIES: ${CERES_LIBRARIES}")


include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/monolidar_fusion
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

############################
## read source code files ##
############################

file(GLOB_RECURSE PROJECT_HEADER_FILES_INC RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "include/*.h" "include/*.hpp")
file(GLOB PROJECT_SOURCE_FILES_INC RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "src/*.h" "src/*.hpp")
file(GLOB PROJECT_SOURCE_FILES_SRC RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "src/*.cpp")

message(STATUS "CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "PROJECT_SOURCE_FILES_SRC: ${PROJECT_SOURCE_FILES_SRC}")

if (PROJECT_SOURCE_FILES_SRC)
    set(LIBRARY_NAME ${PROJECT_NAME})
endif()

###################################
## Eigen
#add_definitions(-DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

###########
## Build ##
###########
# Declare a cpp library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED
    ${PROJECT_SOURCE_FILES_SRC}
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

#############
## Examples ##
#############
add_executable(${PROJECT_NAME}_node 
    examples/monolidar_depth_extraction.cc examples/ORBextractor.cc
)
target_link_libraries(${PROJECT_NAME}_node  ${PROJECT_NAME} )