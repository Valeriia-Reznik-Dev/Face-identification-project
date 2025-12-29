# Face Detection, Recognition and Tracking

C++ application using OpenCV and DLib for face detection, recognition and tracking in video files.

## Requirements

- C++17 compiler (g++ or clang++)
- CMake 3.10 or higher
- OpenCV 4.x
- DLib (optional, but recommended for best results)

## Building

### Step 1: Create build directory

From the project root directory:

```bash
mkdir -p build
cd build
```

### Step 2: Configure with CMake

```bash
cmake -DCMAKE_BUILD_TYPE=Release ../src
```

### Step 3: Compile

```bash
make -j4
```

After successful build, the following executables will be created in `build/`:
- `face_app` - main application for face detection, tracking and recognition
- `metrics` - evaluation tool for computing detection/tracking/recognition metrics
- `recognition` - standalone face recognition demo

## Running

### Basic face detection and tracking

```bash
cd build
./face_app ../dataset/test/videos/person_01_v1.mp4
```

### With face recognition

```bash
cd build
./face_app ../dataset/test/videos/person_01_v1.mp4 \
           --train ../dataset/train \
           --detector dlib_profile \
           --tracker kcf
```

### Computing metrics

```bash
cd build
./metrics ../dataset/test/videos/person_01_v1.mp4 \
          ../dataset/test/annotations/person_01_v1.txt \
          --detector dlib_profile \
          --tracker kcf
```

### Automated testing (optional)

For automated testing on all videos, use the provided script:

```bash
# From project root
./build_and_test.sh
```

The script:
- Builds the project automatically
- Runs metrics on all test videos (person_*)
- Runs metrics on all unknown videos (unknown_*)
- Saves results to `build/results/`

The script works on macOS/Linux. On Windows, use the manual commands above.

## Command-line options

### face_app options:
- `--detector <type>`: haar, haar_profile, lbp, lbp_profile, dlib, dlib_profile (default: haar)
- `--tracker <type>`: kcf, csrt, kcf_fast (default: kcf)
- `--train <dir>`: training directory for recognition
- `--load-descriptors <file>`: load descriptors from file
- `--threshold <value>`: recognition threshold (default: 0.3)
- `--headless`: run without GUI
- `--annotations <file>`: use ground truth annotations
- `--verbose`: verbose output

### metrics options:
- `--detector <type>`: detector type (default: haar)
- `--tracker <type>`: tracker type (default: kcf)
- `--recognition <dir_or_file>`: evaluate recognition metrics
- `--threshold <value>`: recognition threshold (default: 0.3)
- `--iou <value>`: IoU threshold for matching (default: 0.5)
- `--verbose`: detailed per-frame output

## Dataset structure

```
dataset/
├── train/              # Training images (one folder per person)
│   ├── person_01/
│   ├── person_02/
│   └── ...
├── test/
│   ├── videos/        # Test video files
│   ├── annotations/    # Ground truth annotations (Frame: label x y w h)
│   └── images/        # Extracted frames from videos (for submission)
└── unknown_videos/    # Videos with unknown persons (for open-set testing)
    ├── videos/        # Video files with faces not in training set
    └── annotations/   # Annotations (all faces labeled as "unknown")
```

## Annotation format

Annotations are text files with one line per face. Two formats are supported:

Format 1:
```
Frame 0: person_01 100 150 80 100
Frame 10: person_01 105 155 75 95
Frame 20: unknown 200 200 60 80
```

Format 2 (CSV):
```
0,person_01,100,150,80,100
10,person_01,105,155,75,95
20,unknown,200,200,60,80
```

Where coordinates are: frame_id, label, x, y, width, height.

## Results

### Detection + Tracking:
- **TPR (True Positive Rate / Recall)**: percentage of correctly detected faces
- **FPR (False Positive Rate)**: percentage of false detections
- **FNR (False Negative Rate)**: percentage of missed faces
- **Precision**: accuracy of detections

The metrics tool outputs:
- Detection+Tracking: TPR, FPR, FNR
- Recognition: Accuracy, TPR/FPR/FNR for known and unknown persons (open-set protocol)

### Testing with unknown videos

For open-set recognition testing, use videos from `dataset/unknown_videos/` where all faces are labeled as "unknown" in annotations. These videos contain faces that are NOT in the training set. The system should correctly identify these as unknown faces.

Example:
```bash
cd build
./metrics ../dataset/unknown_videos/videos/unknown_01.mp4 \
          ../dataset/unknown_videos/annotations/unknown_01.txt \
          --detector dlib_profile \
          --recognition ../dataset/train
```

The `build_and_test.sh` script automatically processes both test videos (known persons) and unknown videos.

## Implementation details

- Detector: `dlib_profile` (DLib frontal + Haar profile cascades)
- Tracker: `kcf`
- Detection interval: every 2-10 frames
- Recognition: HOG features, cosine distance, k-NN classifier

