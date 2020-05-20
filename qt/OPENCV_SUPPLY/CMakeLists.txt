cmake_minimum_required(VERSION 3.5)

project(OPENCV_SUPPLY LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(OpenCV_DIR "C:/OpenCV-MinGW-Build-OpenCV-4.0.1-x64")
set(OpenCV_INCLUDE_DIRS  "C:/OpenCV-MinGW-Build-OpenCV-4.0.1-x64/include" "C:/OpenCV-MinGW-Build-OpenCV-4.0.1-x64/include/opencv2")
set(OpenCV_LIBS "C:/OpenCV-MinGW-Build-OpenCV-4.0.1-x64/x64/mingw/bin")

include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package( OpenCV REQUIRED )
link_directories(${OpenCV_LIBS})

find_package(Qt5 COMPONENTS Core REQUIRED)

add_library(OPENCV_SUPPLY SHARED
  OPENCV_SUPPLY_global.h
  opencv_supply.cpp
  opencv_supply.h
)

target_link_libraries(OPENCV_SUPPLY  ${OpenCV_LIBS} )
target_link_libraries(OPENCV_SUPPLY  Qt5::Core)

target_compile_definitions(OPENCV_SUPPLY PRIVATE OPENCV_SUPPLY_LIBRARY)

