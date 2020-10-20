cmake_minimum_required(VERSION 3.5)

project(CommonUse LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



set(OpenCV_DIR "D:/wj/OpenCV-MinGW-Build-OpenCV-4.1.0-x64")
set(OpenCV_INCLUDE_DIRS  ${OpenCV_DIR}/include ${OpenCV_DIR}/include/opencv2)
set(OpenCV_LIBS  ${OpenCV_DIR}/x64/mingw/bin)

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


include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package( OpenCV REQUIRED )
link_directories(${OpenCV_LIBS})



file(GLOB_RECURSE TEST_INCLUDE_FILES ./TEST_INCLUDE/*.h )
INCLUDE_DIRECTORIES(./TEST_INCLUDE)


aux_source_directory(./TEST_CPP TEST_SOURCE_DIR)

#这个命令将把该目录下及所有子文件夹内的所有后缀为.cpp的文件的路径，全部放入TEST_SOURCE_FILES这个变量中
#file(GLOB_RECURSE TEST_SOURCE_FILES ${TEST_SOURCE_DIR}/*.CPP )



if(ANDROID)
  add_library(CommonUse SHARED
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
  )
else()
  add_executable(CommonUse
    main.cpp
#    main1.cpp
#    main3.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    ${TEST_INCLUDE_FILES} #为了能编辑，如果说只是用的话 include_directory就够了
    ${TEST_SOURCE_DIR}
  )
endif()

target_link_libraries(CommonUse PRIVATE Qt5::Widgets)
target_link_libraries(CommonUse PUBLIC ${OpenCV_LIBS})
