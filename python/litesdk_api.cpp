#include <iostream>
#include <opencv2/opencv.hpp>
#include "detection.hpp"
#include <fstream>
#include <memory>
#include <vector>

using std::string;
using std::vector;
using std::shared_ptr;


#define LITE_FACEDET_NAME "facedet.mnn"

static shared_ptr<string> sp_cnn_models_{nullptr};

class LiteSdk {
public:
    LiteSdk() = default;
    virtual ~LiteSdk() = default;
    static std::map<std::string, std::unique_ptr<LiteSdk> > sp_sdk_map;
    static LiteSdk* getInstance(const char* id, bool needCreate = false) {
        if (id == nullptr) {
            return nullptr;
        }
        std::string key(id);
        auto find = sp_sdk_map.find(key);
        if (find != sp_sdk_map.end()) {
            return find->second.get();
        }
        if (needCreate) {
            const auto& result = sp_sdk_map.emplace(std::make_pair(id, std::make_unique<LiteSdk>()));
            return result.second ? result.first->second.get() : nullptr;
        } else {
            return nullptr;
        }
    }
    static bool releaseInstance(const char* id) {
        if (id == nullptr) {
            return false;
        }
        std::string key(id);
        auto find = sp_sdk_map.find(key);
        if (find != sp_sdk_map.end()) {
            sp_sdk_map.erase(find);
            return true;
        }
        return false;
    }

public:
    shared_ptr<FaceDetector> sp_unique_;
};
std::map<std::string, std::unique_ptr<LiteSdk> > LiteSdk::sp_sdk_map;


int litesdk_set_models(const char* path,const char* id) {
	if (path == nullptr) {
		return -1;
	}
	string model_path(path);
	if (!model_path.empty()) {
		sp_cnn_models_ = std::make_shared<string>(model_path);
	} else {
		return -1;
	}
	return 0;
}
int litesdk_init(const char* id) {
	if (sp_cnn_models_->empty()) {
        return -1;
    }
	auto facedetPath = *(sp_cnn_models_.get()) + "/" + LITE_FACEDET_NAME;
	// create sdk instance with id
    auto sdk = LiteSdk::getInstance(id, true);
    if (sdk == nullptr) {
        return -1;
    }
    auto& sp_unique_ = sdk->sp_unique_;
	sp_unique_ = std::make_shared<FaceDetector>(facedetPath);
	return 0;
}

int litesdk_face_analysis(lite_image_t image, const char* id,lite_context_t* context) {
	if (image.data == nullptr || image.width <= 0 || image.height <= 0) {
        printf("#### image nullptr: %s\n",id);
        return -1;
    }
	assert(image.stride / image.width == 3);

	if (!context) {
        printf("#### context nullptr: %s\n",id);
        return -1;
    }
	
	auto sdk = LiteSdk::getInstance(id);
    if (sdk == nullptr) {
        printf("#### LiteSdk::getInstance nullptr: %s\n",id);
		context->status = -1;
        return -1;
    }

	std::vector<BoxfWithLandmarks> detected_boxes_kps;
	cv::Mat cv_im(image.height, image.width, CV_8UC3, image.data, image.stride);
    auto& sp_unique_ = sdk->sp_unique_;
	if(sp_unique_ == nullptr) {
		printf("#### sp_unique_ nullptr: %s\n",id);
	}

	sp_unique_->detect(cv_im, detected_boxes_kps);
	
	float max_area = 0;
	BoxfWithLandmarks max_face;
	for(auto& box_kps : detected_boxes_kps) {
		float area = box_kps.box.area();
		if(area > max_area) {
			max_area = area;
			max_face = box_kps;
		}
	}

	context->region.left = max_face.box.x1;
	context->region.top = max_face.box.y1;
	context->region.right = max_face.box.x2;
	context->region.bottom = max_face.box.y2;

	for(int i=0;i<max_face.landmarks.points.size();++i) {
		context->landmark[i].x = max_face.landmarks.points[i].x;
		context->landmark[i].y = max_face.landmarks.points[i].y;
	}

	context->status = 0;

	return 0;
}

int litesdk_release(const char* id) {
    auto result = LiteSdk::releaseInstance(id);
    return result ? 0 : -1;
}