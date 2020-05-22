cmake_minimum_required(VERSION 3.5)

project(OPENCV_SUPPLY_32BIT LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



set(OpenCV_DIR "C:/OpenCV-MinGW-Build-OpenCV-4.1.0_x86")
set(OpenCV_INCLUDE_DIRS  "C:/OpenCV-MinGW-Build-OpenCV-4.1.0_x86/include" "C:/OpenCV-MinGW-Build-OpenCV-4.1.0_x86/include/opencv2/opencv2")
set(OpenCV_LIBS "C:/OpenCV-MinGW-Build-OpenCV-4.1.0_x86/x86/mingw/bin")

include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package( OpenCV REQUIRED )
link_directories(${OpenCV_LIBS})

find_package(Qt5 COMPONENTS Core REQUIRED)

add_library(OPENCV_SUPPLY_32BIT SHARED
  OPENCV_SUPPLY_32BIT_global.h
  opencv_supply_32bit.cpp
  opencv_supply_32bit.h
)

target_link_libraries(OPENCV_SUPPLY_32BIT  Qt5::Core)
target_link_libraries(OPENCV_SUPPLY_32BIT  ${OpenCV_LIBS} )
target_compile_definitions(OPENCV_SUPPLY_32BIT PRIVATE OPENCV_SUPPLY_32BIT_LIBRARY)
