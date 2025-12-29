#include "face_common.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <map>
#include <string>

namespace {

void process_video_with_recognition(const std::string& video_path,
                                    const face::HogFaceRecognizer& recognizer,
                                    bool visualize) {
    cv::CascadeClassifier face_cascade;
    const std::string cascade_path = face::find_cascade_file("haarcascade_frontalface_default.xml");
    if (!face_cascade.load(cascade_path)) {
        std::cerr << "[ERROR] Cannot load Haar cascade: " << cascade_path << std::endl;
        return;
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open video: " << video_path << std::endl;
        return;
    }

    cv::Mat frame;
    cv::Mat gray;

    std::map<std::string, int> counts;
    int total_faces = 0;
    int frame_id = 0;

    while (cap.read(frame)) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(80, 80));

        for (const auto& face : faces) {
            const cv::Rect r = face::clamp_rect(face, gray.cols, gray.rows);
            if (r.width <= 1 || r.height <= 1) continue;

            total_faces++;
            auto result = recognizer.recognize(gray(r));
            std::string label = result.first;
            double distance = result.second;
            counts[label]++;

            if (visualize) {
                const cv::Scalar color = (label == "unknown") ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                cv::rectangle(frame, r, color, 2);

                std::string text = label + " (" + std::to_string(distance).substr(0, 4) + ")";
                cv::putText(frame, text, cv::Point(r.x, std::max(0, r.y - 10)), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            }
        }

        if (visualize) {
            cv::putText(frame, "Frame: " + std::to_string(frame_id), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(255, 255, 255), 2);
            cv::imshow("Recognition", frame);
            if ((char)cv::waitKey(30) == 27) break;
        }

        frame_id++;
        if (frame_id % 30 == 0) {
            std::cout << "[INFO] Frames processed: " << frame_id << std::endl;
        }
    }

    cap.release();
    if (visualize) {
        cv::destroyAllWindows();
    }

    std::cout << "\n[RESULT] Recognition summary\n";
    std::cout << "  Total faces detected: " << total_faces << std::endl;
    std::cout << "  Per-label counts:" << std::endl;
    for (const auto& pair : counts) {
        std::string label = pair.first;
        int count = pair.second;
        const double pct = (total_faces > 0) ? (100.0 * count / total_faces) : 0.0;
        std::cout << "    " << label << ": " << count << " (" << pct << "%)" << std::endl;
    }
}

}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./recognition <video_path> <train_dir> [threshold] [--visualize] "
                     "[--save-descriptors <file>] [--load-descriptors <file>]"
                  << std::endl;
        return 1;
    }

    const std::string video_path = argv[1];
    const std::string train_dir = argv[2];

    double threshold = 0.3;
    bool visualize = false;
    std::string save_descriptors_file;
    std::string load_descriptors_file;

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--visualize") {
            visualize = true;
        } else if (arg == "--save-descriptors" && i + 1 < argc) {
            save_descriptors_file = argv[++i];
        } else if (arg == "--load-descriptors" && i + 1 < argc) {
            load_descriptors_file = argv[++i];
        } else {
            try {
                threshold = std::stod(arg);
            } catch (...) {
            }
        }
    }

    std::cout << "[INFO] Recognition (HOG + cosine distance)\n";
    std::cout << "  Video:     " << video_path << "\n";
    std::cout << "  Train dir: " << train_dir << "\n";
    std::cout << "  Threshold: " << threshold << "\n";
    std::cout << "  Visualize: " << (visualize ? "yes" : "no") << std::endl;

    face::HogFaceRecognizer recognizer(threshold);

    if (!load_descriptors_file.empty()) {
        if (!recognizer.load_descriptors(load_descriptors_file)) {
            std::cerr << "[ERROR] Failed to load descriptors: " << load_descriptors_file << std::endl;
            return 1;
        }
    } else {
        if (!recognizer.train(train_dir, save_descriptors_file)) {
            std::cerr << "[ERROR] Training failed: " << train_dir << std::endl;
            return 1;
        }
    }

    process_video_with_recognition(video_path, recognizer, visualize);
    return 0;
}

