cmake_minimum_required(VERSION 3.5)

project(MyStart LANGUAGES CXX)

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
message(${OpenCV_LIBS})

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check http://doc.qt.io/qt-5/deployment-android.html for more information.
# They need to be set before the find_package(Qt5 ...) call.

#if(ANDROID)
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
#    if (ANDROID_ABI STREQUAL "armeabi-v7a")
#        set(ANDROID_EXTRA_LIBS
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
#    endif()
#endif()

find_package(Qt5 COMPONENTS Widgets REQUIRED)

if(ANDROID)
  add_library(MyStart SHARED
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
  )
else()
  add_executable(MyStart
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    afterprocessslot.h
    ui_mainwindow.h
    imgprocess.h
    imgprocess.cpp
  )
endif()

target_link_libraries(MyStart  Qt5::Widgets)
target_link_libraries(MyStart ${OpenCV_LIBS} )
