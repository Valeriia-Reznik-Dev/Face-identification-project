#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace face {

namespace fs = std::filesystem;

enum class AnnotationFormat {
    XYWH,
    XYXY,
};

struct HaarDetectParams {
    double scaleFactor = 1.02;
    int minNeighbors = 2;
    int minSize = 30;
    double nmsIou = 0.35;
    int maxDetections = 25;
};

inline std::string find_cascade_file(const std::string& filename) {
    const std::vector<std::string> candidates = {
        filename,
        "data/" + filename,
        "../" + filename,
        "../data/" + filename,
        "../../" + filename,
        "../../data/" + filename,
        "/opt/homebrew/share/opencv4/haarcascades/" + filename,
        "/opt/homebrew/share/opencv4/lbpcascades/" + filename,
        "/usr/local/share/opencv4/haarcascades/" + filename,
        "/usr/local/share/opencv4/lbpcascades/" + filename,
        "/usr/share/opencv4/haarcascades/" + filename,
        "/usr/share/opencv4/lbpcascades/" + filename,
    };

    for (size_t i = 0; i < candidates.size(); i++) {
        if (fs::exists(candidates[i])) {
            return candidates[i];
        }
    }
    return filename;
}

struct ParsedAnnotation {
    int frame_id = -1;
    cv::Rect bbox;
    std::string label;
};

inline bool parse_annotation_line(const std::string& line,
                                  ParsedAnnotation& out,
                                  AnnotationFormat format = AnnotationFormat::XYWH) {
    std::string s = line;
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    if (s.empty()) return false;


    if (s.find(',') != std::string::npos) {
        std::vector<std::string> parts;
        std::string cur;
        std::istringstream iss(s);
        while (std::getline(iss, cur, ',')) {
            cur.erase(cur.begin(), std::find_if(cur.begin(), cur.end(), not_space));
            cur.erase(std::find_if(cur.rbegin(), cur.rend(), not_space).base(), cur.end());
            parts.push_back(cur);
        }

        if (parts.size() < 6) return false;

        try {
            out.frame_id = std::stoi(parts[0]);
            out.label = parts[1];
            const int x = std::stoi(parts[2]);
            const int y = std::stoi(parts[3]);
            const int a = std::stoi(parts[4]);
            const int b = std::stoi(parts[5]);
            if (format == AnnotationFormat::XYXY) {
                out.bbox = cv::Rect(x, y, a - x, b - y);
            } else {
                out.bbox = cv::Rect(x, y, a, b);
            }
            return out.bbox.width > 0 && out.bbox.height > 0;
        } catch (...) {
            return false;
        }
    }


    if (s.rfind("Frame ", 0) == 0) {
        const size_t colon_pos = s.find(':');
        if (colon_pos == std::string::npos) return false;

        std::istringstream fss(s.substr(0, colon_pos));
        std::string frame_word;
        int frame_id = -1;
        if (!(fss >> frame_word >> frame_id)) return false;

        std::istringstream iss(s.substr(colon_pos + 1));
        std::string label;
        int x, y, a, b;
        if (!(iss >> label >> x >> y >> a >> b)) return false;

        out.frame_id = frame_id;
        out.label = label;
        if (format == AnnotationFormat::XYXY) {
            out.bbox = cv::Rect(x, y, a - x, b - y);
        } else {
            out.bbox = cv::Rect(x, y, a, b);
        }
        return out.bbox.width > 0 && out.bbox.height > 0;
    }


    {
        std::istringstream iss(s);
        int frame_id, x, y, a, b;
        std::string label;
        if (!(iss >> frame_id >> x >> y >> a >> b >> label)) return false;
        out.frame_id = frame_id;
        out.label = label;
        if (format == AnnotationFormat::XYXY) {
            out.bbox = cv::Rect(x, y, a - x, b - y);
        } else {
            out.bbox = cv::Rect(x, y, a, b);
        }
        return out.bbox.width > 0 && out.bbox.height > 0;
    }
}

inline cv::Rect clamp_rect(const cv::Rect& r, int w, int h) {
    return r & cv::Rect(0, 0, w, h);
}

inline double rect_iou(const cv::Rect& a, const cv::Rect& b) {
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.width, b.x + b.width);
    const int y2 = std::min(a.y + a.height, b.y + b.height);
    if (x2 <= x1 || y2 <= y1) return 0.0;

    const int inter = (x2 - x1) * (y2 - y1);
    const int area_a = a.width * a.height;
    const int area_b = b.width * b.height;
    const int uni = area_a + area_b - inter;
    return (uni > 0) ? static_cast<double>(inter) / static_cast<double>(uni) : 0.0;
}

inline std::vector<cv::Rect> nms_rects(std::vector<cv::Rect> rects, double iou_threshold) {
    if (iou_threshold <= 0.0 || rects.size() <= 1) return rects;

    std::sort(rects.begin(), rects.end(), [](const cv::Rect& a, const cv::Rect& b) {
        return a.area() > b.area();
    });

    std::vector<cv::Rect> keep;

    for (size_t i = 0; i < rects.size(); i++) {
        bool suppress = false;
        for (size_t j = 0; j < keep.size(); j++) {
            if (rect_iou(rects[i], keep[j]) >= iou_threshold) {
                suppress = true;
                break;
            }
        }
        if (!suppress) keep.push_back(rects[i]);
    }
    return keep;
}

inline std::vector<cv::Rect> postprocess_faces(std::vector<cv::Rect> faces,
                                               int img_w,
                                               int img_h,
                                               const HaarDetectParams& params) {
    std::vector<cv::Rect> out;

    for (size_t i = 0; i < faces.size(); i++) {
        cv::Rect r = faces[i];
        r = clamp_rect(r, img_w, img_h);
        if (r.width <= 1 || r.height <= 1) continue;

        double ratio = (double)r.width / (double)r.height;
        if (ratio < 0.5 || ratio > 2.0) continue;

        out.push_back(r);
    }

    out = nms_rects(out, params.nmsIou);

    if (params.maxDetections > 0 && static_cast<int>(out.size()) > params.maxDetections) {
        std::sort(out.begin(), out.end(), [](const cv::Rect& a, const cv::Rect& b) {
            return a.area() > b.area();
        });
        out.resize(params.maxDetections);
    }

    return out;
}

inline std::vector<cv::Rect> detect_faces_haar(const cv::Mat& gray,
                                               cv::CascadeClassifier& cascade,
                                               const HaarDetectParams& params) {
    std::vector<cv::Rect> faces;
    cascade.detectMultiScale(
        gray,
        faces,
        params.scaleFactor,
        params.minNeighbors,
        0,
        cv::Size(params.minSize, params.minSize)
    );
    return postprocess_faces(faces, gray.cols, gray.rows, params);
}

inline std::vector<cv::Rect> detect_faces_haar_profile(const cv::Mat& gray,
                                                       cv::CascadeClassifier& frontal,
                                                       cv::CascadeClassifier& profile,
                                                       const HaarDetectParams& params,
                                                       bool detect_profile_on_flipped = true) {
    std::vector<cv::Rect> all;

    std::vector<cv::Rect> tmp;
    frontal.detectMultiScale(
        gray,
        tmp,
        params.scaleFactor,
        params.minNeighbors,
        0,
        cv::Size(params.minSize, params.minSize)
    );
    all.insert(all.end(), tmp.begin(), tmp.end());

    tmp.clear();
    profile.detectMultiScale(
        gray,
        tmp,
        params.scaleFactor,
        params.minNeighbors,
        0,
        cv::Size(params.minSize, params.minSize)
    );
    all.insert(all.end(), tmp.begin(), tmp.end());

    if (detect_profile_on_flipped) {
        cv::Mat flipped;
        cv::flip(gray, flipped, 1);
        tmp.clear();
        profile.detectMultiScale(
            flipped,
            tmp,
            params.scaleFactor,
            params.minNeighbors,
            0,
            cv::Size(params.minSize, params.minSize)
        );
        for (auto& r : tmp) {
            r.x = gray.cols - r.x - r.width;
            all.push_back(r);
        }
    }

    return postprocess_faces(all, gray.cols, gray.rows, params);
}

class HogFaceRecognizer {
public:
    explicit HogFaceRecognizer(double threshold = 0.3) : threshold_(threshold) {
        hog_ = cv::HOGDescriptor(
            cv::Size(kFaceSize, kFaceSize),
            cv::Size(16, 16),
            cv::Size(8, 8),
            cv::Size(8, 8),
            9
        );
    }

    void set_threshold(double threshold) { threshold_ = threshold; }

    bool empty() const { return templates_.empty(); }

    bool train(const std::string& train_dir, const std::string& save_file = {}) {
        if (!fs::exists(train_dir) || !fs::is_directory(train_dir)) {
            return false;
        }

        templates_.clear();

        for (const auto& entry : fs::directory_iterator(train_dir)) {
            if (!entry.is_directory()) continue;
            std::string person_id = entry.path().filename().string();

            std::vector<cv::Mat> features;
            for (const auto& img_entry : fs::directory_iterator(entry.path())) {
                std::string ext = img_entry.path().extension().string();
                if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

                std::string img_path = img_entry.path().string();
                cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
                if (img.empty()) continue;
                if (img.rows < 40 || img.cols < 40) continue;

                cv::Mat f = extract_features(img);
                if (!f.empty()) features.push_back(f);
            }

            if (!features.empty()) {
                templates_[person_id] = average(features);
            }
        }

        if (!save_file.empty()) {
            save_descriptors(save_file);
        }

        return !templates_.empty();
    }

    bool save_descriptors(const std::string& output_file) const {
        std::ofstream file(output_file);
        if (!file.is_open()) return false;

        for (auto it = templates_.begin(); it != templates_.end(); ++it) {
            file << it->first << ' ' << it->second.cols;
            for (int i = 0; i < it->second.cols; ++i) {
                file << ' ' << it->second.at<float>(0, i);
            }
            file << '\n';
        }
        return true;
    }

    bool load_descriptors(const std::string& input_file) {
        std::ifstream file(input_file);
        if (!file.is_open()) return false;

        templates_.clear();
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::istringstream iss(line);
            std::string person_id;
            int num_features = 0;
            if (!(iss >> person_id >> num_features) || num_features <= 0) continue;

            std::vector<float> values(num_features);
            for (int i = 0; i < num_features; ++i) {
                if (!(iss >> values[i])) {
                    values.clear();
                    break;
                }
            }
            if (values.empty()) continue;

            cv::Mat m(values, true);
            templates_[person_id] = m.reshape(1, 1);
        }

        return !templates_.empty();
    }

    std::pair<std::string, double> recognize(const cv::Mat& face_gray) const {
        if (templates_.empty()) return {"unknown", 1.0};

        cv::Mat features = extract_features(face_gray);
        if (features.empty()) return {"unknown", 1.0};

        double best_dist = 1.0;
        double second_best = 1.0;
        std::string best_id = "unknown";

        for (auto it = templates_.begin(); it != templates_.end(); ++it) {
            double dist = cosine_distance(features, it->second);
            if (dist < best_dist) {
                second_best = best_dist;
                best_dist = dist;
                best_id = it->first;
            } else if (dist < second_best) {
                second_best = dist;
            }
        }

        if (best_dist > threshold_) return {"unknown", best_dist};

        if (second_best < 1.0) {
            const double margin = second_best - best_dist;
            if (margin < 0.05 && best_dist > threshold_ * 0.8) {
                return {"unknown", best_dist};
            }
        }

        return {best_id, best_dist};
    }

private:
    static constexpr int kFaceSize = 128;

    cv::Mat extract_features(const cv::Mat& face_gray) const {
        if (face_gray.empty()) return {};

        cv::Mat gray;
        if (face_gray.channels() == 1) {
            gray = face_gray;
        } else {
            cv::cvtColor(face_gray, gray, cv::COLOR_BGR2GRAY);
        }

        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(kFaceSize, kFaceSize));
        cv::equalizeHist(resized, resized);

        std::vector<float> desc;
        hog_.compute(resized, desc, cv::Size(0, 0), cv::Size(0, 0));

        cv::Mat f(desc, true);
        f = f.reshape(1, 1);

        const double norm = cv::norm(f);
        if (norm > 0.0) {
            f = f / norm;
        }
        return f;
    }

    static cv::Mat average(const std::vector<cv::Mat>& v) {
        if (v.empty()) return {};
        cv::Mat sum = cv::Mat::zeros(1, v[0].cols, CV_32F);
        for (size_t i = 0; i < v.size(); i++) {
            sum += v[i];
        }
        cv::Mat avg = sum / static_cast<double>(v.size());
        const double norm = cv::norm(avg);
        if (norm > 0.0) {
            avg = avg / norm;
        }
        return avg;
    }

    static double cosine_distance(const cv::Mat& a, const cv::Mat& b) {
        const double na = cv::norm(a);
        const double nb = cv::norm(b);
        if (na == 0.0 || nb == 0.0) return 1.0;
        const double sim = a.dot(b) / (na * nb);
        return 1.0 - sim;
    }

    cv::HOGDescriptor hog_;
    std::map<std::string, cv::Mat> templates_;
    double threshold_ = 0.3;
};

}
