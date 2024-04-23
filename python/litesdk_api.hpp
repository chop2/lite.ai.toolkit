
#ifndef __LITE_AI_HPP__
#define __LITE_AI_HPP__


#ifdef _MSC_VER
#ifdef COMPILING_DLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif
#else
#ifdef COMPILING_DLL
#define DLLEXPORT __attribute__((visibility("default")))
#else
#define DLLEXPORT
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif


#define SP_FACE_FTR_DIM                       512

typedef struct lite_image_t {
    lite_image_t() {
        data = nullptr;
        width = height = stride = 0;
        time_stamp = 0;
    }
    unsigned char* data = nullptr;
    int width = 720;
    int height = 1280;
    int stride = 4 * 720;
    double time_stamp = .0;
} lite_image_t;

typedef struct lite_point_f32_t {
    float x = .0f;
    float y = .0f;
} lite_point_f32_t;

typedef struct lite_region_int32_t {
    int left = 0;
    int top = 0;
    int right = 0;
    int bottom = 0;
} lite_region_int32_t;


typedef struct lite_context_t {
    // face region
    lite_region_int32_t region;
    // face score
    float score = .0f;
    // face landmark
    lite_point_f32_t landmark[106];
    // landmark visible
    float visibility[106];
    // face recognition feature
    float feature[SP_FACE_FTR_DIM];
    // face live score
    float face_live = -1;

    // face angle
    float pitch = .0f;
    float yaw = .0f;
    float roll = .0f;

    int face_id = 0;
    int block = 0;
    int gender = 0;

    // app status
    int status = 0;
} lite_context_t, *p_lite_context_t;

DLLEXPORT int litesdk_set_models(const char* modeldir,const char* id);
DLLEXPORT int litesdk_init(const char* id);
DLLEXPORT int litesdk_face_analysis(lite_image_t image, const char* id,lite_context_t* context);
DLLEXPORT int litesdk_release(const char* id);

#ifdef __cplusplus
}
#endif

#endif