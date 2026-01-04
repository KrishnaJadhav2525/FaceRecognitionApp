cmake_minimum_required(VERSION 3.15)
project(FaceRecognitionApp)

set(OpenCV_DIR "D:/opencv/build/install")
find_package(OpenCV REQUIRED)

add_executable(train_app src/train.cpp)
add_executable(recognize_app src/recognize.cpp)

target_link_libraries(train_app ${OpenCV_LIBS})
target_link_libraries(recognize_app ${OpenCV_LIBS})

set_target_properties(train_app recognize_app PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
