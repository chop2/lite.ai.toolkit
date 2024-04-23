#ifdef _PYTHON_API
#include "litesdk_api.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>
#include <pybind11/embed.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(pylitesdk,m) {

    py::class_<lite_image_t>(m, "lite_image_t")
        .def(py::init<>())
        .def_readwrite("width", &lite_image_t::width)
        .def_readwrite("height", &lite_image_t::height)
        .def_readwrite("stride", &lite_image_t::stride)
        .def_readwrite("time_stamp", &lite_image_t::time_stamp)
        .def_property("data",
            [](lite_image_t& img) {
                return py::array_t<uint8_t>(img.height * img.stride);
            },
            [](lite_image_t& img,py::array_t<uint8_t> data) {
                img.data = data.mutable_data();
            }
            );

    py::class_<lite_point_f32_t>(m, "lite_point_f32_t")
        .def(py::init<>())
        .def_readwrite("x", &lite_point_f32_t::x)
        .def_readwrite("y", &lite_point_f32_t::y);

    py::class_<lite_region_int32_t>(m, "lite_region_int32_t")
        .def(py::init<>())
        .def_readwrite("left", &lite_region_int32_t::left)
        .def_readwrite("top", &lite_region_int32_t::top)
        .def_readwrite("right", &lite_region_int32_t::right)
        .def_readwrite("bottom", &lite_region_int32_t::bottom);

    py::class_<lite_context_t>(m, "lite_context_t")
        .def(py::init<>())        
        .def_readwrite("region", &lite_context_t::region)
        .def_readwrite("score", &lite_context_t::score)
        .def_property("landmark", [](lite_context_t &p)->pybind11::array
        {
            auto dtype = pybind11::dtype(pybind11::format_descriptor<float>::format());
            return pybind11::array(dtype, { 106*2 }, { sizeof(float) }, p.landmark, nullptr);
        }, [](lite_context_t& p) {})

        .def_property("visibility", [](lite_context_t &p)->pybind11::array 
        {
            auto dtype = pybind11::dtype(pybind11::format_descriptor<float>::format());
            return pybind11::array(dtype, { 106 }, { sizeof(float) }, p.visibility, nullptr);
        }, [](lite_context_t& p) {})

        .def_property("feature", [](lite_context_t &p)->pybind11::array 
        {
            auto dtype = pybind11::dtype(pybind11::format_descriptor<float>::format());
            return pybind11::array(dtype, { SP_FACE_FTR_DIM }, { sizeof(float) }, p.feature, nullptr);
        }, [](lite_context_t& p) {})

        .def_readwrite("face_live", &lite_context_t::face_live)

        .def_readwrite("pitch", &lite_context_t::pitch)
        .def_readwrite("yaw", &lite_context_t::yaw)
        .def_readwrite("roll", &lite_context_t::roll)
        .def_readwrite("face_id", &lite_context_t::face_id)
        .def_readwrite("block", &lite_context_t::block)
        .def_readwrite("gender", &lite_context_t::gender)
        .def_readwrite("status", &lite_context_t::status);
        
    m.def("litesdk_set_models", &litesdk_set_models, "Set model path");
    m.def("litesdk_init", &litesdk_init, "Initialize SDK");
    m.def("litesdk_face_analysis", &litesdk_face_analysis, "Face analysis");
    m.def("litesdk_release", &litesdk_release, "Release SDK");  

    // m.def("sp_reset", &sp_reset, "Reset results");
    // m.def("sp_update_config", &sp_update_config, "Update config");
    // m.def("sp_commit", &sp_commit, "Detect faces");
    // m.def("sp_get_num_context", &sp_get_num_context, "Get number of faces");
    // m.def("sp_get_context_by_id", &sp_get_context_by_id, "Get face by id");
    // m.def("sp_calc_feature_dist", &sp_calc_feature_dist, "Calculate feature distance");
    // m.def("sp_get_alignface_by_id", &sp_get_alignface_by_id, "Get recognition aligned face");
    // m.def("sp_reco_quality_check",&sp_reco_quality_check,"Check face quality");
    // m.def("sp_get_face_embeding",&sp_get_face_embeding,"Get face embeding by aligned image");

    
/**
 * Code for python call
 * 1.初始化并设置模型路径:
 ```python
    import pysmartpose as sp
    sp.set_models("/path/to/models")
 * ```
 * 2.创建一个句柄并初始化
 ```python
    handle = "face3d"
    sp.init(config, handle)
 ```
 * 3.进行人脸检测:
```python
    image = get_image() # 获取图像
    sp.commit(image, handle)
```
 * 4.获取结果:
 ```python
    num_faces = sp.get_num_context(handle)
    for i in range(num_faces):
    context = sp.Context()
    sp.get_context_by_id(context, i, handle)
    print(context.region)
    # 等等
 ```
 * 5.释放资源:
 ```
    sp.release(handle)
 ```
*/

}

#endif