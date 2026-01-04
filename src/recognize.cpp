#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace cv::face;
using namespace std;

int main() {
    // ---------- PATHS ----------
    string basePath = "D:/Users/GooodForNothing/Desktop/FaceRecognitionApp/data";
    string cascadePath = basePath + "/haarcascade_frontalface_default.xml";
    string modelPath   = basePath + "/face_trained.yml";

    // ---------- LABEL → NAME ----------
    map<int, string> labelNames;
    labelNames[1] = "Krishna";
    labelNames[2] = "Sandu";
    labelNames[3] = "Daleet";
    labelNames[4] = "Tweleve";

    // ---------- Load Cascade ----------
    CascadeClassifier faceCascade;
    if (!faceCascade.load(cascadePath)) {
        cout << "Error loading Haar cascade\n";
        return -1;
    }

    // ---------- Load Model ----------
    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    model->read(modelPath);

    // ---------- Open Camera ----------
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Camera not opened\n";
        return -1;
    }

    Mat frame, gray;
    cout << "Recognition started (ESC to exit)\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 5);

        for (auto &f : faces) {
            Mat faceROI = gray(f);
            resize(faceROI, faceROI, Size(200, 200));

            int label;
            double confidence;
            model->predict(faceROI, label, confidence);

            string text = "Unknown";
            if (confidence < 80 && labelNames.count(label)) {
                text = labelNames[label];
            }

            rectangle(frame, f, Scalar(0,255,0), 2);
            putText(frame, text,
                    Point(f.x, f.y - 10),
                    FONT_HERSHEY_SIMPLEX,
                    0.8,
                    Scalar(0,255,0),
                    2);
        }

        imshow("Face Recognition", frame);
        if (waitKey(10) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
