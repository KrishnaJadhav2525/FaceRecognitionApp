========================================================
REAL-TIME MULTI-USER FACE RECOGNITION SYSTEM (LBPH)
========================================================

Author  : Krishna Pravin Jadhav
Language: C++
Library : OpenCV (with contrib)
Platform: Windows

--------------------------------------------------------
1. PROJECT OVERVIEW
--------------------------------------------------------
This project implements a real-time multi-user face recognition
system using C++ and OpenCV. The system uses the LBPH
(Local Binary Patterns Histogram) algorithm for face recognition
and Haar Cascade for face detection.

Users can train their faces using a webcam and later recognize
them in real time. The project follows a clean and extensible
software design and is suitable for academic, portfolio, and
learning purposes.

--------------------------------------------------------
2. FEATURES
--------------------------------------------------------
- Real-time face detection using Haar Cascade
- Multi-user face recognition using LBPH
- Webcam-based face training
- Clean dataset organization
- CMake-based build system
- Privacy-safe (no face images included in repository)

--------------------------------------------------------
3. TECHNOLOGIES USED
--------------------------------------------------------
- C++
- OpenCV 4.x
- OpenCV Contrib (face module)
- Haar Cascade Classifier
- LBPH Face Recognizer
- CMake
- Visual Studio (Windows)

--------------------------------------------------------
4. PROJECT STRUCTURE
--------------------------------------------------------
FaceRecognitionApp/

│-- CMakeLists.txt
│-- README.txt
│-- .gitignore
│
├── src/
│   ├── train.cpp        -> Face capture and training
│   └── recognize.cpp   -> Real-time face recognition
│
├── data/
│   ├── haarcascade_frontalface_default.xml
│   └── dataset/
│       └── .gitkeep
│
└── LICENSE (optional)

NOTE:
- Face images and trained models are generated at runtime
- These files are not included in the GitHub repository

--------------------------------------------------------
5. PREREQUISITES
--------------------------------------------------------
Before building the project, install the following:

1. Git
2. CMake (version 3.15 or higher)
3. Visual Studio 2019 or 2022
   - Install "Desktop development with C++"
4. OpenCV built with contrib modules

--------------------------------------------------------
6. OPENCV INSTALLATION (REQUIRED)
--------------------------------------------------------

STEP 1: Download OpenCV Source
- Download OpenCV source code
- Download opencv_contrib (same version)

Example:
opencv-4.x.x
opencv_contrib-4.x.x

STEP 2: Configure with CMake GUI
- Source Directory: path/to/opencv
- Build Directory : path/to/opencv/build
- Generator       : Visual Studio
- Platform        : x64

Set:
OPENCV_EXTRA_MODULES_PATH = path/to/opencv_contrib/modules

Click Configure -> Generate

STEP 3: Build and Install
- Open the generated Visual Studio solution
- Build ALL_BUILD
- Build INSTALL

This creates:
opencv/build/install/

STEP 4: Add OpenCV to PATH (Windows)
Add the following to system PATH:
opencv/build/install/x64/vc17/bin

--------------------------------------------------------
7. BUILDING THE PROJECT
--------------------------------------------------------
Clone the repository:

git clone https://github.com/<your-username>/FaceRecognitionApp.git
cd FaceRecognitionApp

Build using CMake:

mkdir build
cd build
cmake ..
cmake --build . --config Release

This generates:
Release/train_app.exe
Release/recognize_app.exe

--------------------------------------------------------
8. RUNNING THE APPLICATION
--------------------------------------------------------

1) TRAIN A NEW USER
-------------------
Run:
Release/train_app.exe

What happens:
- Webcam opens
- Face images are captured
- Images are stored in data/dataset/user_X
- LBPH model is trained and saved

2) RECOGNIZE FACES
------------------
Run:
Release/recognize_app.exe

What happens:
- Webcam opens
- Faces are detected
- Trained users are recognized in real time

--------------------------------------------------------
9. IMPORTANT NOTES
--------------------------------------------------------
- Face images are not included in the repository
- Users must train their own faces locally
- Do not delete the Haar cascade file
- Build files and executables should not be committed

--------------------------------------------------------
10. COMMON ISSUES & SOLUTIONS
--------------------------------------------------------

Problem: Haar cascade not found
Solution:
Ensure the file exists:
data/haarcascade_frontalface_default.xml

Problem: OpenCV DLL error
Solution:
Ensure OpenCV bin directory is added to PATH

Problem: Camera not opening
Solution:
Check webcam permissions or camera index

--------------------------------------------------------
11. FUTURE IMPROVEMENTS
--------------------------------------------------------
- CNN-based face recognition (FaceNet / ArcFace)
- Improved pose and lighting robustness

--------------------------------------------------------
12. LICENSE
--------------------------------------------------------
This project is open-source and can be released
under the MIT License (optional).

========================================================
END OF README
========================================================
