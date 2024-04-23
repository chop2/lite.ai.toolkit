#include "lite/lite.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "litesdk_api.hpp"

using lite::types::BoxfWithLandmarks;
using lite::mnn::cv::face::detect::SCRFD;

class FaceDetector {
    public:
        FaceDetector(const std::string& modelpth);
        ~FaceDetector(){ if (detector) delete detector; };

    void detect(const cv::Mat &mat, std::vector<BoxfWithLandmarks> &detected_boxes_kps,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 400);
    private:
        SCRFD* detector = nullptr;
};