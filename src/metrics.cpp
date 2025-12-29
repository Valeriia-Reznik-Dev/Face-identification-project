#include "face_common.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

#ifdef HAVE_DLIB
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#endif

enum DetectorType { HAAR, HAAR_PROFILE, LBP, LBP_PROFILE, DLIB, DLIB_PROFILE };

enum class TrackerType {
    KCF,
    CSRT,
};

static const char* tracker_name(TrackerType t) {
    switch (t) {
        case TrackerType::KCF: return "KCF";
        case TrackerType::CSRT: return "CSRT";
    }
    return "KCF";
}

static cv::Ptr<cv::Tracker> create_tracker(TrackerType t) {
    if (t == TrackerType::CSRT) return cv::TrackerCSRT::create();
    return cv::TrackerKCF::create();
}

struct GtPadParams {
    double padUp = 0.0;
    double padDown = 0.0;
    double padSide = 0.0;
};

static cv::Rect apply_gt_padding(const cv::Rect& r, int img_w, int img_h, const GtPadParams& pad) {
    int x = r.x - (int)std::round(r.width * pad.padSide);
    int y = r.y - (int)std::round(r.height * pad.padUp);
    int w = r.width + (int)std::round(r.width * pad.padSide * 2.0);
    int h = r.height + (int)std::round(r.height * (pad.padUp + pad.padDown));
    return face::clamp_rect(cv::Rect(x, y, w, h), img_w, img_h);
}

using Annotation = face::ParsedAnnotation;

struct Metrics {
    int tp = 0;
    int fp = 0;
    int fn = 0;

    double tpr() const {
        int total_positives = tp + fn;
        return (total_positives > 0) ? (double)tp / total_positives : 0.0;
    }

    double fpr() const {
        int total = tp + fp;
        return (total > 0) ? (double)fp / total : 0.0;
    }

    double fnr() const {
        int total_positives = tp + fn;
        return (total_positives > 0) ? (double)fn / total_positives : 0.0;
    }

    double precision() const {
        int total_detections = tp + fp;
        return (total_detections > 0) ? (double)tp / total_detections : 0.0;
    }

    double recall() const {
        return tpr();
    }
};


struct RecognitionMetrics {
    int tp_known = 0;
    int fp_known = 0;
    int fn_known = 0;
    int tp_unknown = 0;
    int fp_unknown = 0;
    int fn_unknown = 0;

    double tpr_known() const {
        int total = tp_known + fn_known;
        return (total > 0) ? (double)tp_known / total : 0.0;
    }

    double fpr_known() const {
        int total = tp_known + fp_known;
        return (total > 0) ? (double)fp_known / total : 0.0;
    }

    double fnr_known() const {
        int total = tp_known + fn_known;
        return (total > 0) ? (double)fn_known / total : 0.0;
    }

    double tpr_unknown() const {
        int total = tp_unknown + fn_unknown;
        return (total > 0) ? (double)tp_unknown / total : 0.0;
    }

    double fpr_unknown() const {
        int total = tp_unknown + fp_unknown;
        return (total > 0) ? (double)fp_unknown / total : 0.0;
    }

    double fnr_unknown() const {
        int total = tp_unknown + fn_unknown;
        return (total > 0) ? (double)fn_unknown / total : 0.0;
    }

    double accuracy() const {
        int total = tp_known + fp_known + fn_known + tp_unknown + fp_unknown + fn_unknown;
        int correct = tp_known + tp_unknown;
        return (total > 0) ? (double)correct / total : 0.0;
    }
};

double calculate_iou(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);

    if (x2 <= x1 || y2 <= y1) {
        return 0.0;
    }

    int intersection = (x2 - x1) * (y2 - y1);
    int area_a = a.width * a.height;
    int area_b = b.width * b.height;
    int union_area = area_a + area_b - intersection;

    return (union_area > 0) ? (double)intersection / union_area : 0.0;
}

std::vector<Annotation> load_annotations(const std::string& annotation_path) {
    std::vector<Annotation> annotations;
    std::ifstream file(annotation_path);

    if (!file.is_open()) {
        std::cerr << "[WARN] Cannot open annotation file: " << annotation_path << std::endl;
        return annotations;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        Annotation ann;
        if (face::parse_annotation_line(line, ann, face::AnnotationFormat::XYWH)) {
            annotations.push_back(ann);
        }
    }

    return annotations;
}



static std::vector<cv::Rect> detect_faces(const cv::Mat& gray,
                                   DetectorType detector_type,
                                   cv::CascadeClassifier* frontal_cascade,
                                   cv::CascadeClassifier* profile_cascade,
                                   const face::HaarDetectParams& haar_params
#ifdef HAVE_DLIB
                                   , dlib::frontal_face_detector* dlib_detector = nullptr
#endif
                                   ) {
    if ((detector_type == HAAR || detector_type == LBP) && frontal_cascade) {
        return face::detect_faces_haar(gray, *frontal_cascade, haar_params);
    }

    if ((detector_type == HAAR_PROFILE || detector_type == LBP_PROFILE) && frontal_cascade && profile_cascade) {
        return face::detect_faces_haar_profile(gray, *frontal_cascade, *profile_cascade, haar_params);
    }
#ifdef HAVE_DLIB
    if (detector_type == DLIB && dlib_detector) {
        std::vector<cv::Rect> faces;

        cv::Mat small;
        double scale = 0.5;
        cv::resize(gray, small, cv::Size(), scale, scale);


        dlib::cv_image<unsigned char> dlib_img(small);
        std::vector<dlib::rectangle> dlib_faces = (*dlib_detector)(dlib_img);


        for (const auto& dlib_rect : dlib_faces) {
            faces.push_back(cv::Rect(
                (int)(dlib_rect.left() / scale),
                (int)(dlib_rect.top() / scale),
                (int)(dlib_rect.width() / scale),
                (int)(dlib_rect.height() / scale)
            ));
        }
        return faces;
    }


    if (detector_type == DLIB_PROFILE && dlib_detector && frontal_cascade && profile_cascade) {
        std::vector<cv::Rect> all_faces;


        cv::Mat small;
        double scale = 0.5;
        cv::resize(gray, small, cv::Size(), scale, scale);
        dlib::cv_image<unsigned char> dlib_img(small);
        std::vector<dlib::rectangle> dlib_faces = (*dlib_detector)(dlib_img);

        for (const auto& dlib_rect : dlib_faces) {
            all_faces.push_back(cv::Rect(
                (int)(dlib_rect.left() / scale),
                (int)(dlib_rect.top() / scale),
                (int)(dlib_rect.width() / scale),
                (int)(dlib_rect.height() / scale)
            ));
        }


        std::vector<cv::Rect> profile_faces = face::detect_faces_haar_profile(
            gray, *frontal_cascade, *profile_cascade, haar_params);


        all_faces.insert(all_faces.end(), profile_faces.begin(), profile_faces.end());


        double nms_threshold = (haar_params.nmsIou > 0) ? haar_params.nmsIou : 0.4;
        return face::nms_rects(all_faces, nms_threshold);
    }
#endif

    return {};
}

Metrics evaluate_detection(const std::string& video_path,
                          const std::string& annotation_path,
                          DetectorType detector_type = HAAR,
                          TrackerType tracker_type = TrackerType::KCF,
                          int detect_every_n = 0,
                          face::HaarDetectParams haar_params = {},
                          double iou_threshold = 0.5,
                          GtPadParams gt_pad = {},
                          bool verbose = false) {

    std::vector<Annotation> annotations = load_annotations(annotation_path);
    if (annotations.empty()) {
        std::cerr << "[ERROR] No annotations loaded" << std::endl;
        return Metrics();
    }

    std::cout << "[INFO] Loaded annotations: " << annotations.size() << std::endl;


    std::map<int, std::vector<Annotation>> gt_map;
    for (const auto& ann : annotations) {
        gt_map[ann.frame_id].push_back(ann);
    }

    std::cout << "[INFO] Metrics are computed ONLY on annotated frames" << std::endl;
    std::cout << "   Аннотированных кадров: " << gt_map.size() << std::endl;


    DetectorType effective_detector = detector_type;

    cv::CascadeClassifier frontal_cascade;
    cv::CascadeClassifier profile_cascade;
#ifdef HAVE_DLIB
    dlib::frontal_face_detector dlib_detector;
    bool dlib_initialized = false;
#endif


    if (detector_type == HAAR || detector_type == HAAR_PROFILE ||
        detector_type == LBP || detector_type == LBP_PROFILE) {
        const bool wants_profile = (detector_type == HAAR_PROFILE || detector_type == LBP_PROFILE);
        const bool is_lbp = (detector_type == LBP || detector_type == LBP_PROFILE);

        std::string used_frontal;
        if (is_lbp) {
            std::vector<std::string> candidates = {"lbpcascade_frontalface_improved.xml", "lbpcascade_frontalface.xml"};
            bool loaded = false;
            for (size_t j = 0; j < candidates.size(); j++) {
                std::string path = face::find_cascade_file(candidates[j]);
                if (frontal_cascade.load(path)) {
                    used_frontal = path;
                    loaded = true;
                    break;
                }
            }
            if (!loaded) {
                std::cerr << "[ERROR] Cannot load LBP frontal cascade." << std::endl;
                return Metrics();
            }
        } else {
            std::vector<std::string> candidates = {"haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"};
            bool found = false;
            for (size_t j = 0; j < candidates.size(); j++) {
                std::string path = face::find_cascade_file(candidates[j]);
                if (frontal_cascade.load(path)) {
                    used_frontal = path;
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "[ERROR] Cannot load Haar frontal cascade." << std::endl;
                return Metrics();
            }
        }

        std::string used_profile;
        if (wants_profile) {
            const std::string profile_name = is_lbp ? "lbpcascade_profileface.xml" : "haarcascade_profileface.xml";
            const std::string profile_path = face::find_cascade_file(profile_name);
            if (!profile_cascade.load(profile_path)) {
                std::cerr << "[WARN] Cannot load profile cascade: " << profile_path
                          << " (falling back to frontal-only)" << std::endl;
                effective_detector = is_lbp ? LBP : HAAR;
            } else {
                used_profile = profile_path;
            }
        }

        std::cout << "[INFO] Using " << (is_lbp ? "LBP" : "Haar") << " cascade detector";
        if (wants_profile && (effective_detector == HAAR_PROFILE || effective_detector == LBP_PROFILE)) {
            std::cout << " (frontal + profile + flipped profile)";
        }
        std::cout << std::endl;
        std::cout << "   Frontal: " << used_frontal << std::endl;
        if (!used_profile.empty()) {
            std::cout << "   Profile: " << used_profile << std::endl;
        }
        std::cout << "   Cascade params: scaleFactor=" << haar_params.scaleFactor
                  << ", minNeighbors=" << haar_params.minNeighbors
                  << ", minSize=" << haar_params.minSize
                  << ", nmsIou=" << haar_params.nmsIou
                  << ", maxDetections=" << haar_params.maxDetections << std::endl;
    }
#ifdef HAVE_DLIB
    else if (detector_type == DLIB) {
        dlib_detector = dlib::get_frontal_face_detector();
        dlib_initialized = true;
        std::cout << "[INFO] Using dlib frontal face detector" << std::endl;
        std::cout << "       Optimizations: resize 0.5x, detect every 10 frames" << std::endl;
    }
    else if (detector_type == DLIB_PROFILE) {

        dlib_detector = dlib::get_frontal_face_detector();
        dlib_initialized = true;


        std::string used_frontal;
        std::vector<std::string> candidates = {"haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"};
        bool found = false;
        for (size_t j = 0; j < candidates.size(); j++) {
            std::string path = face::find_cascade_file(candidates[j]);
            if (frontal_cascade.load(path)) {
                used_frontal = path;
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "[ERROR] Cannot load Haar frontal cascade for combined detector." << std::endl;
            return Metrics();
        }

        std::string used_profile;
        const std::string profile_path = face::find_cascade_file("haarcascade_profileface.xml");
        if (!profile_cascade.load(profile_path)) {
            std::cerr << "[ERROR] Cannot load profile cascade for combined detector: " << profile_path << std::endl;
            return Metrics();
        }
        used_profile = profile_path;

        std::cout << "[INFO] Using combined detector: dlib (frontal) + Haar profile cascade" << std::endl;
        std::cout << "   dlib: frontal face detector (resize 0.5x)" << std::endl;
        std::cout << "   Haar frontal: " << used_frontal << std::endl;
        std::cout << "   Haar profile: " << used_profile << std::endl;
        std::cout << "   Cascade params: scaleFactor=" << haar_params.scaleFactor
                  << ", minNeighbors=" << haar_params.minNeighbors
                  << ", minSize=" << haar_params.minSize
                  << ", nmsIou=" << haar_params.nmsIou
                  << ", maxDetections=" << haar_params.maxDetections << std::endl;
    }
#endif
    else {
        std::cerr << "[ERROR] Unknown detector type" << std::endl;
        return Metrics();
    }


    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open video" << std::endl;
        return Metrics();
    }

    cv::Mat frame, gray;
    int frame_id = 0;
    Metrics metrics;

    std::vector<cv::Ptr<cv::Tracker>> trackers;
    std::vector<cv::Rect> tracked_boxes;
    std::vector<int> tracker_init_frames;



    const int DETECT_EVERY_N = (detect_every_n > 0) ? detect_every_n : ((effective_detector == DLIB || effective_detector == DLIB_PROFILE) ? 2 : 2);
    const double IOU_THRESHOLD = iou_threshold;

    int annotated_frames_processed = 0;


    std::vector<int> annotated_frame_ids;
    for (const auto& pair : gt_map) {
        annotated_frame_ids.push_back(pair.first);
    }
    std::sort(annotated_frame_ids.begin(), annotated_frame_ids.end());


    int max_frame_to_process = -1;
    if (!annotated_frame_ids.empty()) {


        max_frame_to_process = annotated_frame_ids.back() + DETECT_EVERY_N * 2 + 10;
        std::cout << "[INFO] Optimized: processing up to frame " << max_frame_to_process
                  << " (last annotated: " << annotated_frame_ids.back() << ")" << std::endl;
    }

    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "[INFO] Total frames in video: " << total_frames << std::endl;
    std::cout << "[INFO] Annotated frames: " << annotated_frame_ids.size() << std::endl;
    std::cout << "[INFO] Detect every " << DETECT_EVERY_N << " frames" << std::endl;


    while (true) {
        cap >> frame;
        if (frame.empty()) break;


        if (max_frame_to_process > 0 && frame_id >= max_frame_to_process) {
            std::cout << "[INFO] Reached optimized limit (frame " << max_frame_to_process << ")" << std::endl;
            break;
        }


        bool has_annotations = (gt_map.find(frame_id) != gt_map.end());



        bool should_process = has_annotations;
        if (!should_process && !annotated_frame_ids.empty()) {

            for (int ann_frame : annotated_frame_ids) {
                if (frame_id >= ann_frame - DETECT_EVERY_N && frame_id <= ann_frame + DETECT_EVERY_N * 2) {
                    should_process = true;
                    break;
                }
            }
        }


        if (!should_process) {
            frame_id++;
            continue;
        }


        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        bool do_detection = (frame_id % DETECT_EVERY_N == 0) || trackers.empty();


        if (has_annotations && annotated_frames_processed % 5 == 0 && annotated_frames_processed > 0) {
            std::cout << "   Обработано аннотированных кадров: " << annotated_frames_processed
                      << " / " << annotated_frame_ids.size() << std::endl;
        }

        if (do_detection) {


            trackers.clear();
            tracked_boxes.clear();
            tracker_init_frames.clear();

            std::vector<cv::Rect> faces = detect_faces(
                gray,
                effective_detector,
                (effective_detector == HAAR || effective_detector == HAAR_PROFILE || effective_detector == DLIB_PROFILE) ? &frontal_cascade : nullptr,
                (effective_detector == HAAR_PROFILE || effective_detector == DLIB_PROFILE) ? &profile_cascade : nullptr,
#ifdef HAVE_DLIB
                haar_params,
                ((effective_detector == DLIB || effective_detector == DLIB_PROFILE) && dlib_initialized) ? &dlib_detector : nullptr
#else
                haar_params
#endif
            );


            for (const auto& face : faces) {
                const cv::Rect r = face::clamp_rect(face, frame.cols, frame.rows);
                if (r.width <= 1 || r.height <= 1) continue;

                cv::Ptr<cv::Tracker> tracker = create_tracker(tracker_type);
                if (!tracker) continue;
                tracker->init(frame, r);
                trackers.push_back(tracker);
                tracked_boxes.push_back(r);
                tracker_init_frames.push_back(frame_id);
            }
        } else {


            for (size_t i = 0; i < trackers.size(); ) {
                bool ok = trackers[i]->update(frame, tracked_boxes[i]);
                if (!ok) {
                    trackers.erase(trackers.begin() + (long)i);
                    tracked_boxes.erase(tracked_boxes.begin() + (long)i);
                    tracker_init_frames.erase(tracker_init_frames.begin() + (long)i);
                    continue;
                }

                tracked_boxes[i] &= cv::Rect(0, 0, frame.cols, frame.rows);
                if (tracked_boxes[i].width <= 1 || tracked_boxes[i].height <= 1) {
                    trackers.erase(trackers.begin() + (long)i);
                    tracked_boxes.erase(tracked_boxes.begin() + (long)i);
                    tracker_init_frames.erase(tracker_init_frames.begin() + (long)i);
                    continue;
                }

                ++i;
            }
        }


        if (has_annotations) {
            annotated_frames_processed++;


            const auto& gt_boxes_raw = gt_map[frame_id];
            std::vector<Annotation> gt_boxes = gt_boxes_raw;
            for (auto& gt : gt_boxes) {
                gt.bbox = apply_gt_padding(gt.bbox, gray.cols, gray.rows, gt_pad);
            }

            if (verbose) {
                std::cout << "   Frame " << frame_id << ": GT=" << gt_boxes.size()
                          << " faces, Detected=" << tracked_boxes.size() << " faces" << std::endl;
            }


            int valid_trackers = 0;
            for (size_t i = 0; i < tracked_boxes.size(); ++i) {
                int init_frame = (i < tracker_init_frames.size()) ? tracker_init_frames[i] : frame_id;
                bool init_on_annotated = (gt_map.find(init_frame) != gt_map.end());
                bool init_near_current = (init_frame >= frame_id - DETECT_EVERY_N && init_frame <= frame_id);
                if (init_on_annotated || init_near_current) {
                    valid_trackers++;
                }
            }
            if (valid_trackers > gt_boxes.size() * 2 && !verbose) {

            }




            std::vector<bool> gt_matched(gt_boxes.size(), false);




            for (size_t i = 0; i < tracked_boxes.size(); ++i) {
                int init_frame = (i < tracker_init_frames.size()) ? tracker_init_frames[i] : frame_id;
                bool init_on_annotated = (gt_map.find(init_frame) != gt_map.end());




                bool should_evaluate = init_on_annotated;
                if (!should_evaluate) {

                    if (init_frame >= frame_id - DETECT_EVERY_N && init_frame < frame_id) {
                        should_evaluate = true;
                    }
                }

                if (!should_evaluate) {

                    continue;
                }

                double best_iou = 0.0;
                int best_gt_idx = -1;

                for (size_t j = 0; j < gt_boxes.size(); ++j) {
                    if (gt_matched[j]) continue;

                    double iou = calculate_iou(tracked_boxes[i], gt_boxes[j].bbox);
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_gt_idx = j;
                    }
                }

                if (best_iou >= IOU_THRESHOLD && best_gt_idx >= 0) {
                    metrics.tp++;
                    gt_matched[best_gt_idx] = true;
                    if (verbose) {
                        std::cout << "      TP: IoU=" << best_iou << " (init_frame=" << init_frame << ")" << std::endl;
                    }
                } else {


                    if (init_on_annotated || (init_frame >= frame_id - DETECT_EVERY_N && init_frame < frame_id)) {
                        metrics.fp++;
                        if (verbose) {
                            std::cout << "      FP: best_IoU=" << best_iou << " (init_frame=" << init_frame << ")" << std::endl;
                        }
                    }
                }
            }


            for (size_t i = 0; i < gt_boxes.size(); ++i) {
                if (!gt_matched[i]) {
                    metrics.fn++;
                    if (verbose) {
                        std::cout << "      FN: GT face #" << i << " missed" << std::endl;
                    }
                }
            }
        }

        frame_id++;
    }

    cap.release();

    std::cout << "[INFO] Annotated frames processed: " << annotated_frames_processed << std::endl;

    return metrics;
}


using SimpleFaceRecognizer = face::HogFaceRecognizer;

RecognitionMetrics evaluate_recognition(const std::string& video_path,
                                       const std::string& annotation_path,
                                       const std::string& train_dir_or_descriptors,
                                       double threshold,
                                       bool load_from_file = false,
                                       bool use_gt_boxes_only = true,
                                       face::HaarDetectParams haar_params = {},
                                       double iou_threshold = 0.5,
                                       GtPadParams gt_pad = {}) {
    RecognitionMetrics metrics;


    std::vector<Annotation> annotations = load_annotations(annotation_path);
    if (annotations.empty()) {
        std::cerr << "[ERROR] No annotations loaded" << std::endl;
        return metrics;
    }


    std::map<int, std::vector<Annotation>> gt_map;
    for (const auto& ann : annotations) {
        gt_map[ann.frame_id].push_back(ann);
    }


    SimpleFaceRecognizer recognizer(threshold);
    if (load_from_file) {
        if (!recognizer.load_descriptors(train_dir_or_descriptors)) {
            std::cerr << "[ERROR] Cannot load descriptors" << std::endl;
            return metrics;
        }
    } else {
        if (!recognizer.train(train_dir_or_descriptors)) {
            std::cerr << "[ERROR] Cannot train recognizer" << std::endl;
            return metrics;
        }
    }


    cv::CascadeClassifier face_cascade;
    std::string cascade_path = face::find_cascade_file("haarcascade_frontalface_default.xml");
    if (!face_cascade.load(cascade_path)) {
        std::cerr << "[ERROR] Cannot load Haar cascade" << std::endl;
        return metrics;
    }


    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Cannot open video" << std::endl;
        return metrics;
    }

    cv::Mat frame, gray;
    int frame_id = 0;
    const double IOU_THRESHOLD = iou_threshold;

    std::vector<int> annotated_frame_ids;
    for (const auto& pair : gt_map) {
        annotated_frame_ids.push_back(pair.first);
    }
    std::sort(annotated_frame_ids.begin(), annotated_frame_ids.end());

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        bool has_annotations = (gt_map.find(frame_id) != gt_map.end());
        if (!has_annotations) {
            frame_id++;
            continue;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        const auto& gt_boxes_raw = gt_map[frame_id];
        std::vector<Annotation> gt_boxes = gt_boxes_raw;
        for (auto& gt : gt_boxes) {
            gt.bbox = apply_gt_padding(gt.bbox, gray.cols, gray.rows, gt_pad);
        }

        if (use_gt_boxes_only) {

            for (const auto& gt : gt_boxes) {
                cv::Rect r = face::clamp_rect(gt.bbox, gray.cols, gray.rows);
                if (r.width <= 1 || r.height <= 1) continue;

                cv::Mat face_roi = gray(r);
                auto result = recognizer.recognize(face_roi);
                std::string predicted_label = result.first;
                double distance = result.second;
                (void)distance;
                std::string true_label = gt.label;

                bool true_is_unknown = (true_label == "unknown");
                bool pred_is_unknown = (predicted_label == "unknown");

                if (!true_is_unknown && !pred_is_unknown) {
                    if (predicted_label == true_label) {
                        metrics.tp_known++;
                    } else {
                        metrics.fn_known++;
                        metrics.fp_known++;
                    }
                } else if (true_is_unknown && pred_is_unknown) {
                    metrics.tp_unknown++;
                } else if (!true_is_unknown && pred_is_unknown) {
                    metrics.fp_unknown++;
                    metrics.fn_known++;
                } else if (true_is_unknown && !pred_is_unknown) {
                    metrics.fn_unknown++;
                    metrics.fp_known++;
                }
            }
        } else {

            std::vector<cv::Rect> faces = detect_faces(gray, HAAR, &face_cascade, nullptr, haar_params);

            std::vector<bool> gt_matched(gt_boxes.size(), false);

            for (size_t i = 0; i < faces.size(); ++i) {
                const cv::Rect det = face::clamp_rect(faces[i], gray.cols, gray.rows);
                if (det.width <= 1 || det.height <= 1) continue;

                double best_iou = 0.0;
                int best_gt_idx = -1;

                for (size_t j = 0; j < gt_boxes.size(); ++j) {
                    if (gt_matched[j]) continue;
                    double iou = calculate_iou(det, gt_boxes[j].bbox);
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_gt_idx = (int)j;
                    }
                }

                if (best_iou >= IOU_THRESHOLD && best_gt_idx >= 0) {
                    gt_matched[best_gt_idx] = true;

                    cv::Mat face_roi = gray(det);
                    auto result = recognizer.recognize(face_roi);
                    std::string predicted_label = result.first;
                    double distance = result.second;
                    (void)distance;
                    std::string true_label = gt_boxes[best_gt_idx].label;

                    bool true_is_unknown = (true_label == "unknown");
                    bool pred_is_unknown = (predicted_label == "unknown");

                    if (!true_is_unknown && !pred_is_unknown) {
                        if (predicted_label == true_label) {
                            metrics.tp_known++;
                        } else {
                            metrics.fn_known++;
                            metrics.fp_known++;
                        }
                    } else if (true_is_unknown && pred_is_unknown) {
                        metrics.tp_unknown++;
                    } else if (!true_is_unknown && pred_is_unknown) {
                        metrics.fp_unknown++;
                        metrics.fn_known++;
                    } else if (true_is_unknown && !pred_is_unknown) {
                        metrics.fn_unknown++;
                        metrics.fp_known++;
                    }
                }
            }


            for (size_t i = 0; i < gt_boxes.size(); ++i) {
                if (!gt_matched[i] && gt_boxes[i].label != "unknown") {
                    metrics.fn_known++;
                }
            }
        }

        frame_id++;
        if (frame_id > annotated_frame_ids.back()) break;
    }

    cap.release();
    return metrics;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./metrics <video_path> <annotation_path> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --detector <type>       : haar (default), haar_profile, lbp, lbp_profile, dlib, dlib_profile" << std::endl;
        std::cerr << "  --tracker <type>        : kcf (default), csrt, kcf_fast" << std::endl;
        std::cerr << "  --detect-every <int>    : run detector every N frames (default depends on detector/tracker)" << std::endl;
        std::cerr << "  --recognition <dir_or_file> : evaluate recognition metrics" << std::endl;
        std::cerr << "  --load-descriptors      : if recognition file is descriptors file (not train dir)" << std::endl;
        std::cerr << "  --threshold <value>     : recognition threshold (default 0.3)" << std::endl;
        std::cerr << "  --recognition-e2e       : recognition end-to-end (with face detection); default = on GT boxes" << std::endl;
        std::cerr << "  --haar-scale <value>    : Haar scaleFactor (default 1.05)" << std::endl;
        std::cerr << "  --haar-neighbors <int>  : Haar minNeighbors (default 3)" << std::endl;
        std::cerr << "  --haar-min-size <int>   : Haar minSize in px (default 40)" << std::endl;
        std::cerr << "  --nms-iou <value>       : NMS IoU for Haar detections (default 0.4, 0=off)" << std::endl;
        std::cerr << "  --max-detections <int>  : limit detections per frame (default 10, 0=unlimited)" << std::endl;
        std::cerr << "  --iou <value>           : IoU threshold for matching (default 0.5)" << std::endl;
        std::cerr << "  --gt-pad <up> <down> <side> : expand GT bbox by fractions of h/h/w (default 0 0 0)" << std::endl;
        std::cerr << "  --verbose               : print per-frame matching details" << std::endl;
        return 1;
    }

    std::string video_path = argv[1];
    std::string annotation_path = argv[2];
    TrackerType tracker_type = TrackerType::KCF;
    int detect_every_n = 0;
    bool verbose = false;
    DetectorType detector_type = HAAR;
    bool evaluate_recognition_metrics = false;
    std::string train_dir_or_descriptors;
    bool load_from_file = false;
    double recognition_threshold = 0.3;
    bool recognition_gt_only = true;
    face::HaarDetectParams haar_params;
    double iou_threshold = 0.5;
    GtPadParams gt_pad;


    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tracker" && i + 1 < argc) {
            const std::string t = argv[++i];
            if (t == "csrt" || t == "CSRT") {
                tracker_type = TrackerType::CSRT;
            } else if (t == "kcf_fast" || t == "KCF_FAST") {
                tracker_type = TrackerType::KCF;
                if (detect_every_n == 0) detect_every_n = 3;
            } else {
                tracker_type = TrackerType::KCF;
            }
        } else if (arg == "--detector" && i + 1 < argc) {
            std::string det_type = argv[++i];
            if (det_type == "haar_profile" || det_type == "HAAR_PROFILE") {
                detector_type = HAAR_PROFILE;
            } else if (det_type == "lbp_profile" || det_type == "LBP_PROFILE") {
                detector_type = LBP_PROFILE;
            } else if (det_type == "lbp" || det_type == "LBP") {
                detector_type = LBP;
            } else if (det_type == "dlib" || det_type == "DLIB") {
#ifdef HAVE_DLIB
                detector_type = DLIB;
#else
                std::cerr << "[WARN] dlib not found; falling back to Haar." << std::endl;
                detector_type = HAAR;
#endif
            } else if (det_type == "dlib_profile" || det_type == "DLIB_PROFILE") {
#ifdef HAVE_DLIB
                detector_type = DLIB_PROFILE;
#else
                std::cerr << "[WARN] dlib not found; falling back to haar_profile." << std::endl;
                detector_type = HAAR_PROFILE;
#endif
            } else {
                detector_type = HAAR;
            }
        } else if (arg == "--recognition" && i + 1 < argc) {
            train_dir_or_descriptors = argv[++i];
            evaluate_recognition_metrics = true;
        } else if (arg == "--load-descriptors") {
            load_from_file = true;
        } else if (arg == "--threshold" && i + 1 < argc) {
            recognition_threshold = std::stod(argv[++i]);
        } else if (arg == "--recognition-e2e") {
            recognition_gt_only = false;
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
        } else if (arg == "--iou" && i + 1 < argc) {
            iou_threshold = std::stod(argv[++i]);
        } else if (arg == "--gt-pad" && i + 3 < argc) {
            gt_pad.padUp = std::stod(argv[++i]);
            gt_pad.padDown = std::stod(argv[++i]);
            gt_pad.padSide = std::stod(argv[++i]);
        } else if (arg == "--detect-every" && i + 1 < argc) {
            detect_every_n = std::stoi(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        }
    }


    if (!evaluate_recognition_metrics) {
        const std::string detector_name =
            (detector_type == DLIB_PROFILE) ? "dlib_profile"
            : (detector_type == DLIB) ? "dlib"
            : (detector_type == HAAR_PROFILE) ? "haar_profile"
            : (detector_type == LBP_PROFILE) ? "lbp_profile"
            : (detector_type == LBP) ? "lbp"
            : "haar";
        std::cout << "[INFO] Detection+tracking metrics\n";
        std::cout << "  Detector: " << detector_name << "\n";
        std::cout << "  Tracker:  " << tracker_name(tracker_type) << "\n";
        if (detect_every_n > 0) {
            std::cout << "  Detect every: " << detect_every_n << " frames\n";
        }
        std::cout << std::endl;

        Metrics metrics = evaluate_detection(
            video_path,
            annotation_path,
            detector_type,
            tracker_type,
            detect_every_n,
            haar_params,
            iou_threshold,
            gt_pad,
            verbose
        );

        std::cout << "\n[RESULT] Detection summary\n";
        std::cout << "  NOTE: metrics are computed ONLY on annotated frames\n";
        std::cout << "   TP (True Positive):  " << metrics.tp << std::endl;
        std::cout << "   FP (False Positive): " << metrics.fp << std::endl;
        std::cout << "   FN (False Negative): " << metrics.fn << std::endl;
        std::cout << "\n[RESULT] Detection metrics\n";
        std::cout << "   TPR (Recall):        " << (metrics.tpr() * 100.0) << "%" << std::endl;
        std::cout << "   FPR (FP/(TP+FP)):    " << (metrics.fpr() * 100.0) << "%" << std::endl;
        std::cout << "   FNR:                 " << (metrics.fnr() * 100.0) << "%" << std::endl;
        std::cout << "   Precision:           " << (metrics.precision() * 100.0) << "%" << std::endl;
        std::cout << "   Recall:              " << (metrics.recall() * 100.0) << "%" << std::endl;
    }


    if (evaluate_recognition_metrics) {
        if (train_dir_or_descriptors.empty()) {
            std::cerr << "[ERROR] Missing --recognition <dir_or_file>" << std::endl;
            return 1;
        }

        std::cout << "\n[INFO] Recognition metrics (open-set protocol)\n";
        std::cout << "  Threshold: " << recognition_threshold << "\n";
        std::cout << "  Source: " << (load_from_file ? "descriptors file" : "train directory") << std::endl;

        RecognitionMetrics rec_metrics = evaluate_recognition(
            video_path, annotation_path, train_dir_or_descriptors,
            recognition_threshold, load_from_file, recognition_gt_only, haar_params, iou_threshold, gt_pad
        );

        std::cout << "\n[RESULT] Recognition counts" << std::endl;
        std::cout << "  Known:" << std::endl;
        std::cout << "      TP (правильно распознан): " << rec_metrics.tp_known << std::endl;
        std::cout << "      FP (неизвестный как известный): " << rec_metrics.fp_known << std::endl;
        std::cout << "      FN (не распознан): " << rec_metrics.fn_known << std::endl;
        std::cout << "  Unknown:" << std::endl;
        std::cout << "      TP (правильно как unknown): " << rec_metrics.tp_unknown << std::endl;
        std::cout << "      FP (известный как unknown): " << rec_metrics.fp_unknown << std::endl;
        std::cout << "      FN (unknown как известный): " << rec_metrics.fn_unknown << std::endl;

        std::cout << "\n[RESULT] Recognition metrics" << std::endl;
        std::cout << "   Accuracy:            " << (rec_metrics.accuracy() * 100.0) << "%" << std::endl;
        std::cout << "   Known:" << std::endl;
        std::cout << "      TPR (Recall):     " << (rec_metrics.tpr_known() * 100.0) << "%" << std::endl;
        std::cout << "      FPR:              " << (rec_metrics.fpr_known() * 100.0) << "%" << std::endl;
        std::cout << "      FNR:              " << (rec_metrics.fnr_known() * 100.0) << "%" << std::endl;
        std::cout << "   Unknown:" << std::endl;
        std::cout << "      TPR (Recall):     " << (rec_metrics.tpr_unknown() * 100.0) << "%" << std::endl;
        std::cout << "      FPR:              " << (rec_metrics.fpr_unknown() * 100.0) << "%" << std::endl;
        std::cout << "      FNR:              " << (rec_metrics.fnr_unknown() * 100.0) << "%" << std::endl;
    }

    return 0;
}
