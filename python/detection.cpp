#include "detection.hpp"
FaceDetector::FaceDetector(const std::string& modelpth) {
    detector = new SCRFD(modelpth,1);
}

void FaceDetector::detect(const cv::Mat &mat, std::vector<BoxfWithLandmarks> &detected_boxes_kps,
                float score_threshold, float iou_threshold,
                unsigned int topk) 
{
    if(detector == nullptr) {
        printf("detector is nullptr\n");
        return;
    }
    detector->detect(mat,detected_boxes_kps,score_threshold,iou_threshold,topk );
}


// int main(int argc,char** argv) {
//     std::string onnx_path = "yolov5s.onnx";
//     std::string test_img_path = "";
//     std::string save_img_path = "";

//     // auto* yolov5 = new lite::cv::detection::YoloV5(onnx_path);
//     auto* yolov5 = new lite::mnn::cv::detection::YoloV5(onnx_path);
//     std::vector<lite::types::Boxf> detected_boxes;
//     cv::Mat img_bgr = cv::imread(test_img_path);
//     yolov5->detect(img_bgr, detected_boxes);

//     lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);
//     cv::imwrite(save_img_path, img_bgr);
//     delete yolov5;
//     return 0;
// }
