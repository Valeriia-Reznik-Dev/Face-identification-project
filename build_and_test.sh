#!/bin/bash
# Build and test script for Face Recognition project with parallel processing

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
SRC_DIR="${PROJECT_ROOT}/src"

echo "=== Building Face Recognition Project ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Build directory: ${BUILD_DIR}"
echo "Build type: Release"
echo ""

# Clean build directory
if [ -d "${BUILD_DIR}" ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with Release mode
echo "Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE=Release "${SRC_DIR}" >/dev/null 2>&1

# Build all targets
echo "Building all targets..."
make -j$(sysctl -n hw.ncpu) >/dev/null 2>&1

echo "=== Build completed successfully ==="
echo ""

# Check if executables exist
if [ ! -f "${BUILD_DIR}/metrics" ]; then
    echo "ERROR: metrics not found!"
    exit 1
fi

TEST_VIDEOS_DIR="${PROJECT_ROOT}/dataset/test/videos"
TEST_ANNOTATIONS_DIR="${PROJECT_ROOT}/dataset/test/annotations"
UNKNOWN_VIDEOS_DIR="${PROJECT_ROOT}/dataset/unknown_videos/videos"
UNKNOWN_ANNOTATIONS_DIR="${PROJECT_ROOT}/dataset/unknown_videos/annotations"
DETECTOR="dlib_profile"
RESULTS_DIR="${BUILD_DIR}/results"
mkdir -p "${RESULTS_DIR}"

# Function to process a single video with progress output
process_video() {
    local VIDEO="$1"
    local ANNOTATION="$2"
    local VIDEO_NAME="$3"
    local RESULT_FILE="${RESULTS_DIR}/${VIDEO_NAME}.txt"
    local START_TIME=$(date +%s)
    
    echo "[$(date '+%H:%M:%S')] Начало обработки: ${VIDEO_NAME}"
    
    "${BUILD_DIR}/metrics" \
        "${VIDEO}" \
        "${ANNOTATION}" \
        --detector "${DETECTOR}" > "${RESULT_FILE}" 2>&1
    
    local EXIT_CODE=$?
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    local MIN=$((DURATION / 60))
    local SEC=$((DURATION % 60))
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        # Extract metrics with improved parsing
        local TP=$(grep "TP (True Positive):" "${RESULT_FILE}" | awk '{print $4}')
        local FP=$(grep "FP (False Positive):" "${RESULT_FILE}" | awk '{print $4}')
        local FN=$(grep "FN (False Negative):" "${RESULT_FILE}" | awk '{print $4}')
        # TPR is in field 3: "TPR (Recall): 99.4286%"
        local TPR=$(grep "TPR (Recall):" "${RESULT_FILE}" | awk '{print $3}' | sed 's/%//')
        
        # Fallback parsing if TPR not found
        if [ -z "${TPR}" ] || [ "${TPR}" = "" ]; then
            TPR=$(grep "^   Recall:" "${RESULT_FILE}" | awk '{print $3}' | sed 's/%//')
        fi
        
        # Validate extracted values
        if [ -n "${TP}" ] && [ -n "${TPR}" ] && [ "${TP}" != "" ] && [ "${TPR}" != "" ]; then
            echo "[$(date '+%H:%M:%S')] ✅ ${VIDEO_NAME}: завершено за ${MIN}м ${SEC}с | TPR=${TPR}%, TP=${TP}, FP=${FP}, FN=${FN}"
            echo "${VIDEO_NAME}|${TPR}|${TP}|${FP}|${FN}" >> "${RESULTS_DIR}/all_results.txt"
            return 0
        else
            echo "[$(date '+%H:%M:%S')] ⚠️  ${VIDEO_NAME}: завершено за ${MIN}м ${SEC}с, но не удалось распарсить результаты"
            echo "${VIDEO_NAME}|FAILED|0|0|0" >> "${RESULTS_DIR}/all_results.txt"
            return 1
        fi
    else
        echo "[$(date '+%H:%M:%S')] ❌ ${VIDEO_NAME}: ОШИБКА (exit code ${EXIT_CODE}) за ${MIN}м ${SEC}с"
        echo "${VIDEO_NAME}|FAILED|0|0|0" >> "${RESULTS_DIR}/all_results.txt"
        return 1
    fi
}

# Function to calculate and display summary
show_summary() {
    local DATASET_NAME="$1"
    local RESULTS_FILE="${RESULTS_DIR}/all_results.txt"
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    SUMMARY: ${DATASET_NAME}                                                      ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Calculate totals
    local TOTAL_TP=0
    local TOTAL_FP=0
    local TOTAL_FN=0
    local SUCCESS_COUNT=0
    local FAILED_COUNT=0
    local TPR_SUM=0
    
    if [ -f "${RESULTS_FILE}" ]; then
        while IFS='|' read -r name tpr tp fp fn; do
            # Filter by dataset prefix
            if [[ "${name}" =~ ^${DATASET_NAME} ]]; then
                if [ "${tpr}" != "FAILED" ] && [ -n "${tp}" ] && [ "${tp}" != "" ]; then
                    TOTAL_TP=$((TOTAL_TP + tp))
                    TOTAL_FP=$((TOTAL_FP + fp))
                    TOTAL_FN=$((TOTAL_FN + fn))
                    TPR_SUM=$(echo "${TPR_SUM} + ${tpr}" | bc -l 2>/dev/null || echo "${TPR_SUM} + ${tpr}" | awk '{print $1 + $3}')
                    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                else
                    FAILED_COUNT=$((FAILED_COUNT + 1))
                fi
            fi
        done < "${RESULTS_FILE}"
    fi
    
    if [ ${SUCCESS_COUNT} -gt 0 ]; then
        local AVG_TPR=$(echo "scale=2; ${TPR_SUM} / ${SUCCESS_COUNT}" | bc -l 2>/dev/null || echo "scale=2; ${TPR_SUM} / ${SUCCESS_COUNT}" | awk '{printf "%.2f", $1/$3}')
        local TOTAL_POSITIVES=$((TOTAL_TP + TOTAL_FN))
        local OVERALL_TPR=$(echo "scale=2; ${TOTAL_TP} * 100 / ${TOTAL_POSITIVES}" | bc -l 2>/dev/null || awk "BEGIN {printf \"%.2f\", ${TOTAL_TP} * 100 / ${TOTAL_POSITIVES}}")
        local TOTAL_DETECTIONS=$((TOTAL_TP + TOTAL_FP))
        local OVERALL_PRECISION=$(echo "scale=2; ${TOTAL_TP} * 100 / ${TOTAL_DETECTIONS}" | bc -l 2>/dev/null || awk "BEGIN {printf \"%.2f\", ${TOTAL_TP} * 100 / ${TOTAL_DETECTIONS}}")
        
        printf "┌─────────────────────────────────────────────────────────────────────────────────────┐\n"
        printf "│ Обработано видео:                                          %3d                      │\n" ${SUCCESS_COUNT}
        if [ ${FAILED_COUNT} -gt 0 ]; then
            printf "│ Ошибок:                                                   %3d                      │\n" ${FAILED_COUNT}
        fi
        printf "├─────────────────────────────────────────────────────────────────────────────────────┤\n"
        printf "│ Overall TPR (Recall):                                      %6.2f%%                  │\n" ${OVERALL_TPR}
        printf "│ Average TPR (средний по видео):                            %6.2f%%                  │\n" ${AVG_TPR}
        printf "│ Overall Precision:                                          %6.2f%%                  │\n" ${OVERALL_PRECISION}
        printf "├─────────────────────────────────────────────────────────────────────────────────────┤\n"
        printf "│ Total TP (True Positive):                                  %5d                      │\n" ${TOTAL_TP}
        printf "│ Total FP (False Positive):                                 %5d                      │\n" ${TOTAL_FP}
        printf "│ Total FN (False Negative):                                 %5d                      │\n" ${TOTAL_FN}
        printf "└─────────────────────────────────────────────────────────────────────────────────────┘\n"
        
        echo ""
        echo "Детальные результаты:"
        grep "^${DATASET_NAME}" "${RESULTS_FILE}" 2>/dev/null | while IFS='|' read -r name tpr tp fp fn; do
            if [ "${tpr}" != "FAILED" ]; then
                printf "  ✓ %-20s TPR: %7s%%, TP: %4s, FP: %5s, FN: %3s\n" "${name}" "${tpr}" "${tp}" "${fp}" "${fn}"
            else
                printf "  ✗ %-20s FAILED\n" "${name}"
            fi
        done | sort
    else
        echo "Нет успешно обработанных видео"
    fi
    echo ""
}

# Clear results file
> "${RESULTS_DIR}/all_results.txt"

# Function to process a dataset
process_dataset() {
    local VIDEOS_DIR="$1"
    local ANNOTATIONS_DIR="$2"
    local DATASET_NAME="$3"
    local PREFIX="$4"
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════════════════════════════╗"
    echo "║              ОБРАБОТКА ${DATASET_NAME}                                                          ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Collect videos
    local VIDEO_LIST=()
    local VIDEO_ANNOTATIONS=()
    local VIDEO_NAMES=()
    
    for VIDEO in "${VIDEOS_DIR}"/*.mp4; do
        if [ -f "${VIDEO}" ]; then
            local VIDEO_NAME=$(basename "${VIDEO}" .mp4)
            local ANNOTATION="${ANNOTATIONS_DIR}/${VIDEO_NAME}.txt"
            if [ -f "${ANNOTATION}" ]; then
                VIDEO_LIST+=("${VIDEO}")
                VIDEO_ANNOTATIONS+=("${ANNOTATION}")
                VIDEO_NAMES+=("${VIDEO_NAME}")
            fi
        fi
    done
    
    local TOTAL_VIDEOS=${#VIDEO_LIST[@]}
    if [ ${TOTAL_VIDEOS} -eq 0 ]; then
        echo "Нет видео для обработки в ${DATASET_NAME}"
        return
    fi
    
    echo "Всего видео для обработки: ${TOTAL_VIDEOS}"
    echo "Обработка в параллельном режиме..."
    echo ""
    
    # Process videos in parallel (limit to number of CPU cores)
    local MAX_PARALLEL=$(sysctl -n hw.ncpu)
    local CURRENT=0
    local COMPLETED=0
    local FAILED=0
    
    for i in "${!VIDEO_LIST[@]}"; do
        local VIDEO="${VIDEO_LIST[$i]}"
        local ANNOTATION="${VIDEO_ANNOTATIONS[$i]}"
        local VIDEO_NAME="${VIDEO_NAMES[$i]}"
        
        CURRENT=$((CURRENT + 1))
        echo "[${CURRENT}/${TOTAL_VIDEOS}] Запуск обработки: ${VIDEO_NAME}..."
        
        # Run in background
        process_video "${VIDEO}" "${ANNOTATION}" "${VIDEO_NAME}" &
        
        # Limit parallel jobs
        if [ $((CURRENT % MAX_PARALLEL)) -eq 0 ]; then
            wait  # Wait for batch to complete
        fi
    done
    
    # Wait for all remaining jobs
    wait
    
    # Count completed and failed from results file
    COMPLETED=0
    FAILED=0
    if [ -f "${RESULTS_DIR}/all_results.txt" ]; then
        while IFS='|' read -r name tpr tp fp fn; do
            if [[ "${name}" =~ ^${PREFIX} ]]; then
                if [ "${tpr}" != "FAILED" ] && [ -n "${tp}" ] && [ "${tp}" != "" ]; then
                    COMPLETED=$((COMPLETED + 1))
                else
                    FAILED=$((FAILED + 1))
                fi
            fi
        done < "${RESULTS_DIR}/all_results.txt"
    fi
    
    echo ""
    echo "─────────────────────────────────────────────────────────────────────────────────────────────"
    echo "Обработка ${DATASET_NAME} завершена: успешно ${COMPLETED}, ошибок ${FAILED}"
    echo "─────────────────────────────────────────────────────────────────────────────────────────────"
    
    # Show summary for this dataset
    show_summary "${PREFIX}"
}

# Process test videos (person_*) first
process_dataset "${TEST_VIDEOS_DIR}" "${TEST_ANNOTATIONS_DIR}" "TEST DATASET" "person_"

# Process unknown videos second
process_dataset "${UNKNOWN_VIDEOS_DIR}" "${UNKNOWN_ANNOTATIONS_DIR}" "UNKNOWN DATASET" "unknown_"

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                    ВСЯ ОБРАБОТКА ЗАВЕРШЕНА                                                    ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""
