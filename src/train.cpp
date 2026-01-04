#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <filesystem>
#include <iostream>

using namespace cv;
using namespace cv::face;
using namespace std;
namespace fs = std::filesystem;

int main() {
    
    string basePath = "D:/Users/GooodForNothing/Desktop/FaceRecognitionApp/data";
    string cascadePath = basePath + "/haarcascade_frontalface_default.xml";
    string datasetBase = basePath + "/dataset";
    string modelPath   = basePath + "/face_trained.yml";

    int userId = 4;  
    string userPath = datasetBase + "/user_" + to_string(userId);

    CascadeClassifier faceCascade;
    if (!faceCascade.load(cascadePath)) {
        cout << "Error loading Haar cascade\n";
        return -1;
    }

   
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Camera not opened\n";
        return -1;
    }

    fs::create_directories(userPath);

    Mat frame, gray;
    int count = 0;

    cout << "Collecting images for user_" << userId << endl;

    
    while (count < 50) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 5);

        for (auto &f : faces) {
            Mat faceROI = gray(f);
            resize(faceROI, faceROI, Size(200, 200));

            string filename = userPath + "/img_" + to_string(count) + ".jpg";
            imwrite(filename, faceROI);

            rectangle(frame, f, Scalar(0,255,0), 2);
            count++;
        }

        imshow("Training", frame);
        if (waitKey(100) == 27) break;
    }

    cap.release();
    destroyAllWindows();

    
    vector<Mat> images;
    vector<int> labels;

    for (auto &userDir : fs::directory_iterator(datasetBase)) {
        if (!userDir.is_directory()) continue;

        string folderName = userDir.path().filename().string();
        int label = stoi(folderName.substr(5)); // user_X → X

        for (auto &imgPath : fs::directory_iterator(userDir.path())) {
            Mat img = imread(imgPath.path().string(), IMREAD_GRAYSCALE);
            if (!img.empty()) {
                images.push_back(img);
                labels.push_back(label);
            }
        }
    }

    if (images.empty()) {
        cout << "No images found for training!\n";
        return -1;
    }

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    model->train(images, labels);
    model->save(modelPath);

    cout << "Training complete. Model saved to:\n" << modelPath << endl;
    return 0;
}
