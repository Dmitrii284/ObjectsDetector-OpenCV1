cmake_minimum_required(VERSION 3.26)
project(ObjectsDetector)

set(CMAKE_CXX_STANDARD 23)

# Установка переменной окружения OPENCV_DIR
set(ENV{OPENCV_DIR} "/opt/opencv/release")

set(SOURCES
        main.cpp
)
# Добавление пути, где находятся файлы cmake для OpenCV
set(CMAKE_PREFIX_PATH "/home/dmitriy/opencv-4.x/build" ${CMAKE_PREFIX_PATH})

# Поиск и добавление пакета OpenCV
find_package(OpenCV REQUIRED)

find_package(Boost COMPONENTS system filesystem program_options REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
)
