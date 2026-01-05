cmake_minimum_required(VERSION 3.15)
project(FaceRecognitionApp)

set(OpenCV_DIR "D:/opencv/build/install")
find_package(OpenCV REQUIRED)

add_executable(FaceRecognitionApp src/main.cpp)
target_link_libraries(FaceRecognitionApp ${OpenCV_LIBS})


set_target_properties(FaceRecognitionApp PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
