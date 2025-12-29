#include "face_common.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kDetectEveryN = 2;
constexpr int kDetectEveryNFast = 3;

enum class DetectorType {
    Haar,
    HaarProfile,
    Lbp,
    LbpProfile,
};

DetectorType parse_detector_type(const std::string& s) {
    if (s == "haar_profile" || s == "HAAR_PROFILE") return DetectorType::HaarProfile;
    if (s == "lbp_profile" || s == "LBP_PROFILE") return DetectorType::LbpProfile;
    if (s == "lbp" || s == "LBP") return DetectorType::Lbp;
    return DetectorType::Haar;
}

enum class TrackerType {
    Kcf,
    Csrt,
    KcfFast,
};

TrackerType parse_tracker_type(const std::string& s) {
    if (s == "csrt" || s == "CSRT") return TrackerType::Csrt;
    if (s == "kcf_fast" || s == "KCF_FAST") return TrackerType::KcfFast;
    return TrackerType::Kcf;
}

int detection_interval(TrackerType t) {
    return (t == TrackerType::KcfFast) ? kDetectEveryNFast : kDetectEveryN;
}

std::string tracker_name(TrackerType t) {
    switch (t) {
        case TrackerType::Kcf: return "KCF";
        case TrackerType::Csrt: return "CSRT";
        case TrackerType::KcfFast: return "KCF_FAST";
    }
    return "KCF";
}

cv::Ptr<cv::Tracker> create_tracker(TrackerType t) {
    if (t == TrackerType::Csrt) return cv::TrackerCSRT::create();
    return cv::TrackerKCF::create();
}

void update_trackers(const cv::Mat& frame,
                     std::vector<cv::Ptr<cv::Tracker>>& trackers,
                     std::vector<cv::Rect>& boxes,
                     std::vector<std::pair<std::string, double>>& labels) {
    for (size_t i = 0; i < trackers.size(); ) {
        bool ok = trackers[i]->update(frame, boxes[i]);
        boxes[i] = face::clamp_rect(boxes[i], frame.cols, frame.rows);

        if (!ok || boxes[i].width <= 1 || boxes[i].height <= 1) {
            trackers.erase(trackers.begin() + i);
            boxes.erase(boxes.begin() + i);
            if (i < labels.size()) {
                labels.erase(labels.begin() + i);
            }
            continue;
        }

        ++i;
    }
}

}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./face_app <video_path> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --detector <type>    : haar (default), haar_profile, lbp, lbp_profile" << std::endl;
        std::cerr << "  --tracker <type>     : kcf (default), csrt, kcf_fast" << std::endl;
        std::cerr << "  --train <dir>        : train directory for recognition" << std::endl;
        std::cerr << "  --load-descriptors <file> : load descriptors from file" << std::endl;
        std::cerr << "  --threshold <value>  : recognition threshold (default 0.3)" << std::endl;
        std::cerr << "  --haar-scale <value> : Haar scaleFactor (default 1.05)" << std::endl;
        std::cerr << "  --haar-neighbors <int> : Haar minNeighbors (default 3)" << std::endl;
        std::cerr << "  --haar-min-size <int> : Haar minSize in px (default 40)" << std::endl;
        std::cerr << "  --nms-iou <value>    : NMS IoU for Haar detections (default 0.4, 0=off)" << std::endl;
        std::cerr << "  --max-detections <int> : limit detections per frame (default 10, 0=unlimited)" << std::endl;
        std::cerr << "  --annotations <file> : use annotations from txt file" << std::endl;
        std::cerr << "  --save-annotations <file> : save detections to txt file" << std::endl;
        std::cerr << "  --first-frame-only  : process only first frame" << std::endl;
        std::cerr << "  --headless          : run without displaying window" << std::endl;
        std::cerr << "  --verbose           : print per-frame debug info" << std::endl;
        return 1;
    }

    std::string video_path = argv[1];
    std::string train_dir;
    std::string load_descriptors_file;
    std::string annotations_file;
    std::string save_annotations_file;
    double recognition_threshold = 0.3;
    bool enable_recognition = false;
    bool first_frame_only = false;
    bool use_annotations = false;
    bool save_detections = false;
    bool verbose = false;
    bool headless = false;


    DetectorType detector_type = DetectorType::Haar;
    TrackerType tracker_type = TrackerType::Kcf;
    face::HaarDetectParams haar_params;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--detector" && i + 1 < argc) {
            detector_type = parse_detector_type(argv[++i]);
        } else if (arg == "--tracker" && i + 1 < argc) {
            tracker_type = parse_tracker_type(argv[++i]);
        } else if (arg == "--train" && i + 1 < argc) {
            train_dir = argv[++i];
            enable_recognition = true;
        } else if (arg == "--load-descriptors" && i + 1 < argc) {
            load_descriptors_file = argv[++i];
            enable_recognition = true;
        } else if (arg == "--threshold" && i + 1 < argc) {
            recognition_threshold = std::stod(argv[++i]);
        } else if (arg == "--haar-scale" && i + 1 < argc) {
            haar_params.scaleFactor = std::stod(argv[++i]);
        } else if (arg == "--haar-neighbors" && i + 1 < argc) {
            haar_params.minNeighbors = std::stoi(argv[++i]);
        } else if (arg == "--haar-min-size" && i + 1 < argc) {
            haar_params.minSize = std::stoi(argv[++i]);
        } else if (arg == "--nms-iou" && i + 1 < argc) {
            haar_params.nmsIou = std::stod(argv[++i]);
        } else if (arg == "--max-detections" && i + 1 < argc) {
            haar_params.maxDetections = std::stoi(argv[++i]);
        } else if (arg == "--first-frame-only") {
            first_frame_only = true;
        } else if (arg == "--annotations" && i + 1 < argc) {
            annotations_file = argv[++i];
            use_annotations = true;
        } else if (arg == "--save-annotations" && i + 1 < argc) {
            save_annotations_file = argv[++i];
            save_detections = true;
        } else if (arg == "--headless") {
            headless = true;
        } else if (arg == "--verbose") {
            verbose = true;
        }
    }


    if (!use_annotations && annotations_file.empty()) {
        size_t last_slash = video_path.find_last_of("/");
        size_t last_dot = video_path.find_last_of(".");
        if (last_slash != std::string::npos && last_dot != std::string::npos && last_dot > last_slash) {
            std::string video_dir = video_path.substr(0, last_slash);
            std::string video_basename = video_path.substr(last_slash + 1, last_dot - last_slash - 1);
            std::string auto_annotations = video_dir + "/" + video_basename + ".txt";
            if (fs::exists(auto_annotations)) {
                annotations_file = auto_annotations;
                use_annotations = true;
                std::cout << "[INFO] Found annotations: " << annotations_file << std::endl;
            }
        }
    }


    face::HogFaceRecognizer* recognizer = nullptr;
    face::HogFaceRecognizer recognizer_obj(recognition_threshold);
    if (enable_recognition) {
        recognizer = &recognizer_obj;
        if (!load_descriptors_file.empty()) {
            if (!recognizer->load_descriptors(load_descriptors_file)) {
                std::cerr << "[ERROR] Failed to load descriptors: " << load_descriptors_file << std::endl;
                return 1;
            }
            std::cout << "[INFO] Descriptors loaded: " << load_descriptors_file << std::endl;
        } else if (!train_dir.empty()) {
            if (!recognizer->train(train_dir)) {
                std::cerr << "[ERROR] Failed to train on: " << train_dir << std::endl;
                return 1;
            }
            std::cout << "[INFO] Training done (" << train_dir << ")" << std::endl;
        }
    }


    cv::CascadeClassifier frontal_cascade;
    cv::CascadeClassifier profile_cascade;

    std::string frontal_path;

    const bool is_lbp = (detector_type == DetectorType::Lbp || detector_type == DetectorType::LbpProfile);
    const bool wants_profile = (detector_type == DetectorType::HaarProfile || detector_type == DetectorType::LbpProfile);

    if (is_lbp) {
        std::vector<std::string> candidates = {"lbpcascade_frontalface_improved.xml", "lbpcascade_frontalface.xml"};
        bool loaded = false;
        for (size_t j = 0; j < candidates.size(); j++) {
            std::string path = face::find_cascade_file(candidates[j]);
            if (frontal_cascade.load(path)) {
                frontal_path = path;
                loaded = true;
                break;
            }
        }
        if (!loaded) {
            std::cerr << "[ERROR] Cannot load LBP frontal cascade." << std::endl;
            return 1;
        }
    } else {
        std::vector<std::string> candidates = {"haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"};
        bool found = false;
        for (size_t j = 0; j < candidates.size(); j++) {
            std::string path = face::find_cascade_file(candidates[j]);
            if (frontal_cascade.load(path)) {
                frontal_path = path;
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "[ERROR] Cannot load Haar frontal cascade." << std::endl;
            return 1;
        }
    }

    bool have_profile_cascade = false;
    std::string profile_path;
    if (wants_profile) {
        const std::string profile_name = is_lbp ? "lbpcascade_profileface.xml" : "haarcascade_profileface.xml";
        profile_path = face::find_cascade_file(profile_name);
        have_profile_cascade = profile_cascade.load(profile_path);
        if (!have_profile_cascade) {
            std::cerr << "[WARN] Cannot load profile cascade: " << profile_path
                      << " (falling back to frontal-only)" << std::endl;
            detector_type = is_lbp ? DetectorType::Lbp : DetectorType::Haar;
        }
    }


    std::map<int, std::vector<face::ParsedAnnotation>> annotations_map;
    if (use_annotations && !annotations_file.empty()) {
        std::ifstream ann_file(annotations_file);
        if (ann_file.is_open()) {
            std::string line;
            int loaded_count = 0;
            while (std::getline(ann_file, line)) {
                face::ParsedAnnotation ann;
                if (!face::parse_annotation_line(line, ann, face::AnnotationFormat::XYWH)) continue;
                annotations_map[ann.frame_id].push_back(ann);
                loaded_count++;
            }
            ann_file.close();
            std::cout << "[INFO] Loaded annotations: " << loaded_count << " (" << annotations_file << ")" << std::endl;
        } else {
            std::cerr << "[WARN] Cannot open annotations file: " << annotations_file << std::endl;
            use_annotations = false;
        }
    }

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open video" << std::endl;
        return 1;
    }

    cv::Mat frame, gray;
    int frame_id = 0;

    std::vector<cv::Ptr<cv::Tracker>> trackers;
    std::vector<cv::Rect> tracked_boxes;
    std::vector<std::pair<std::string, double>> tracked_labels;


    std::map<int, std::vector<std::pair<cv::Rect, std::string>>> detections_map;

    const std::string tracker_label = tracker_name(tracker_type);
    const int detect_interval = detection_interval(tracker_type);
    const std::string detector_label =
        (detector_type == DetectorType::HaarProfile && have_profile_cascade) ? "HAAR_PROFILE"
        : (detector_type == DetectorType::LbpProfile && have_profile_cascade) ? "LBP_PROFILE"
        : (detector_type == DetectorType::Lbp) ? "LBP"
        : "HAAR";
    const std::string window_title = "Detection + Tracking (" + tracker_label + ", " + detector_label + ")";

    std::cout << "[INFO] Detector: " << detector_label << " (frontal=" << frontal_path;
    if ((detector_type == DetectorType::HaarProfile || detector_type == DetectorType::LbpProfile) && have_profile_cascade) {
        std::cout << ", profile=" << profile_path;
    }
    std::cout << ")\n";
    std::cout << "[INFO] Cascade params: scaleFactor=" << haar_params.scaleFactor
              << ", minNeighbors=" << haar_params.minNeighbors
              << ", minSize=" << haar_params.minSize
              << ", nmsIou=" << haar_params.nmsIou
              << ", maxDetections=" << haar_params.maxDetections << "\n";
    std::cout << "[INFO] Tracker: " << tracker_label << ", detect every " << detect_interval << " frames" << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        bool do_detection = (frame_id % detect_interval == 0);

        if (do_detection) {

            trackers.clear();
            tracked_boxes.clear();

            std::vector<cv::Rect> faces;
            std::vector<std::string> annotation_names;


            if (use_annotations && annotations_map.find(frame_id) != annotations_map.end()) {
                const std::vector<face::ParsedAnnotation>& frame_anns = annotations_map[frame_id];
                for (size_t j = 0; j < frame_anns.size(); j++) {
                    cv::Rect r = face::clamp_rect(frame_anns[j].bbox, gray.cols, gray.rows);
                    if (r.width <= 1 || r.height <= 1) continue;
                    faces.push_back(r);
                    annotation_names.push_back(frame_anns[j].label);
                }
                if (verbose) {
                    std::cout << "[INFO] Frame " << frame_id << ": using " << faces.size() << " annotated boxes" << std::endl;
                }
            } else {
                if ((detector_type == DetectorType::HaarProfile || detector_type == DetectorType::LbpProfile) && have_profile_cascade) {
                    faces = face::detect_faces_haar_profile(gray, frontal_cascade, profile_cascade, haar_params);
                } else {
                    faces = face::detect_faces_haar(gray, frontal_cascade, haar_params);
                }
            }

            tracked_labels.clear();
            for (size_t i = 0; i < faces.size(); ++i) {
                const auto& face = faces[i];
                cv::Ptr<cv::Tracker> tracker = create_tracker(tracker_type);
                if (tracker) {
                    tracker->init(frame, face);
                    trackers.push_back(tracker);
                    tracked_boxes.push_back(face);

                    if (enable_recognition && recognizer) {
                        const cv::Rect r = face::clamp_rect(face, gray.cols, gray.rows);
                        cv::Mat roi = gray(r);
                        auto result = recognizer->recognize(roi);
                        std::string label = result.first;
                        double distance = result.second;

                        if (distance > recognition_threshold * 1.5) {
                            label = "unknown";
                        }

                        tracked_labels.push_back({label, distance});

                        if (verbose && use_annotations && i < annotation_names.size()) {
                            std::string expected = annotation_names[i];
                            std::cout << "[DEBUG] Frame " << frame_id << ": predicted=" << label
                                      << ", expected=" << expected << ", dist=" << distance << std::endl;
                        } else if (verbose) {
                            std::cout << "[DEBUG] Frame " << frame_id << ": predicted=" << label
                                      << ", dist=" << distance << std::endl;
                        }

                        if (save_detections) {
                            detections_map[frame_id].push_back({face, label});
                        }
                    } else {
                        tracked_labels.push_back({"", 0.0});
                        if (save_detections) {
                            detections_map[frame_id].push_back({face, "unknown"});
                        }
                    }
                }
            }
        } else {

            update_trackers(frame, trackers, tracked_boxes, tracked_labels);
        }


        for (size_t i = 0; i < tracked_boxes.size(); ++i) {
            const auto& box = tracked_boxes[i];
            cv::Scalar color = cv::Scalar(0, 255, 0);


            if (enable_recognition && i < tracked_labels.size() && !tracked_labels[i].first.empty()) {
                const std::pair<std::string, double>& label_pair = tracked_labels[i];
                std::string label = label_pair.first;
                double distance = label_pair.second;

                if (label == "unknown") {
                    color = cv::Scalar(0, 0, 255);
                }
                cv::rectangle(frame, box, color, 2);


                std::string label_text = label;
                if (distance < 1.0) {
                    label_text += " (" + std::to_string(distance).substr(0, 4) + ")";
                }
                cv::putText(
                    frame,
                    label_text,
                    cv::Point(box.x, box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                );
            } else {
                cv::rectangle(frame, box, color, 2);
            }
        }


        std::string info_text = "Frame: " + std::to_string(frame_id) + " [" + tracker_label + ", " + detector_label + "]";
        cv::putText(
            frame,
            info_text,
            cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0, 255, 0),
            2
        );

        if (!headless) {
            cv::imshow(window_title, frame);


            if (first_frame_only && frame_id == 0) {
                std::cout << "[INFO] First frame shown. Press any key to close..." << std::endl;
            }
            int wait_time = first_frame_only ? 0 : 30;
            char key = (char)cv::waitKey(wait_time);
            if (key == 27) break;
        }
        if (first_frame_only) break;

        frame_id++;
    }

    cap.release();


    if (save_detections && !save_annotations_file.empty() && !detections_map.empty()) {
        std::ofstream out_file(save_annotations_file);
        if (out_file.is_open()) {
            int saved_count = 0;

            for (const auto& pair : detections_map) {
                int frame_idx = pair.first;
                const auto& detections = pair.second;
                if (!detections.empty()) {
                    cv::Rect box = detections[0].first;
                    std::string label = detections[0].second;

                    if (box.width >= 50 && box.height >= 50 &&
                        box.width <= 400 && box.height <= 400) {
                        out_file << "Frame " << frame_idx << ": " << label << " "
                                 << box.x << " " << box.y << " " << box.width << " " << box.height << "\n";
                        saved_count++;
                    }
                }
            }
            out_file.close();
            std::cout << "[INFO] Saved " << saved_count << " detections to " << save_annotations_file << std::endl;
        } else {
            std::cerr << "[ERROR] Cannot write: " << save_annotations_file << std::endl;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
