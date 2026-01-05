#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

using namespace cv;
using namespace cv::face;
using namespace std;
namespace fs = std::filesystem;

// ---------- GLOBAL PATHS ----------
string basePath    = "D:/Users/GooodForNothing/Desktop/FaceRecognitionApp/data";
string cascadePath = basePath + "/haarcascade_frontalface_default.xml";
string datasetBase = basePath + "/dataset";
string modelPath   = basePath + "/face_trained.yml";
string labelFile   = basePath + "/labels.txt";

// ---------- TRAIN MODE ----------
void trainUser() {
    // Find next user ID
    int nextUserId = 1;
    for (auto &dir : fs::directory_iterator(datasetBase)) {
        if (dir.is_directory()) {
            string name = dir.path().filename().string();
            if (name.rfind("user_", 0) == 0) {
                int id = stoi(name.substr(5));
                nextUserId = max(nextUserId, id + 1);
            }
        }
    }

    // Ask name
    string userName;
    cout << "Enter name (no spaces): ";
    cin >> userName;

    string userPath = datasetBase + "/user_" + to_string(nextUserId);
    fs::create_directories(userPath);

    // Save label
    ofstream out(labelFile, ios::app);
    out << nextUserId << " " << userName << endl;
    out.close();

    cout << "Registering user_" << nextUserId << " (" << userName << ")\n";

    // Load cascade
    CascadeClassifier faceCascade;
    faceCascade.load(cascadePath);

    VideoCapture cap(0);
    Mat frame, gray;
    int count = 0;

    cout << "Collecting images...\n";

    while (count < 50) {
        cap >> frame;
        if (frame.empty()) continue;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        faceCascade.detectMultiScale(
            gray,
            faces,
            1.1,
            4,
            0,
            Size(80, 80)
        );


        for (auto &f : faces) {
            Mat faceROI = gray(f);
            resize(faceROI, faceROI, Size(200, 200));
            equalizeHist(faceROI, faceROI);

            imwrite(userPath + "/img_" + to_string(count++) + ".jpg", faceROI);
            rectangle(frame, f, Scalar(0,255,0), 2);
        }

        imshow("Train Mode", frame);
        if (waitKey(100) == 27) break;
    }

    cap.release();
    destroyAllWindows();

    // Train model with all users
    vector<Mat> images;
    vector<int> labels;

    for (auto &dir : fs::directory_iterator(datasetBase)) {
        if (!dir.is_directory()) continue;
        int label = stoi(dir.path().filename().string().substr(5));

        for (auto &img : fs::directory_iterator(dir.path())) {
            Mat im = imread(img.path().string(), IMREAD_GRAYSCALE);
            if (!im.empty()) {
                images.push_back(im);
                labels.push_back(label);
            }
        }
    }

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    model->train(images, labels);
    model->save(modelPath);

    cout << "Training complete.\n";
}

// ---------- RECOGNITION MODE ----------
void recognizeFaces() {
    map<int,string> labelNames;
    ifstream in(labelFile);
    int id; string name;
    while (in >> id >> name) labelNames[id] = name;
    in.close();

    CascadeClassifier faceCascade;
    faceCascade.load(cascadePath);

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    model->read(modelPath);

    VideoCapture cap(0);
    Mat frame, gray;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        faceCascade.detectMultiScale(
            gray,
            faces,
            1.1,
            4,
            0,
            Size(80, 80)
        );



        for (auto &f : faces) {
            Mat faceROI = gray(f);
           
            resize(faceROI, faceROI, Size(200, 200));
            equalizeHist(faceROI, faceROI);


            int label; double confidence;
            model->predict(faceROI, label, confidence);

            const double THRESHOLD = 90.0;   // try 60–70 later

            string text = "Unknown";
            if (confidence < THRESHOLD && labelNames.count(label)) {
                text = labelNames[label] + " (" + to_string((int)confidence) + ")";
            }
            else {
                text = "Unknown (" + to_string((int)confidence) + ")";
            }


            rectangle(frame, f, Scalar(0,255,0), 2);
            putText(frame, text, Point(f.x, f.y-10),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);
        }

        imshow("Recognition Mode", frame);
        if (waitKey(10) == 27) break;
    }

    cap.release();
    destroyAllWindows();
}

// ---------- MAIN ----------
int main() {
    while (true) {
        cout << "\n=== FACE RECOGNITION SYSTEM ===\n";
        cout << "T - Train new user\n";
        cout << "R - Recognize faces\n";
        cout << "Q - Quit\n";
        cout << "Choice: ";

        char choice;
        cin >> choice;

        if (choice == 'T' || choice == 't') trainUser();
        else if (choice == 'R' || choice == 'r') recognizeFaces();
        else if (choice == 'Q' || choice == 'q') break;
    }
    return 0;
}
