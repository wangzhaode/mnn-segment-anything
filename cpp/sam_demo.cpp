#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>

#include <cv/cv.hpp>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::CV;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: ./sam_demo.out embed.mnn sam.mnn input.jpg [forwardType] [precision] [thread]\n");
        return 0;
    }
    int thread = 4;
    int precision = 0;
    int forwardType = MNN_FORWARD_CPU;
    if (argc >= 5) {
        forwardType = atoi(argv[4]);
    }
    if (argc >= 6) {
        precision = atoi(argv[5]);
    }
    if (argc >= 7) {
        thread = atoi(argv[6]);
    }
    float mask_threshold = 0;
    MNN::ScheduleConfig sConfig;
    sConfig.type = static_cast<MNNForwardType>(forwardType);
    sConfig.numThread = thread;
    BackendConfig bConfig;
    bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
    sConfig.backendConfig = &bConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(sConfig));
    if(rtmgr == nullptr) {
        MNN_ERROR("Empty RuntimeManger\n");
        return 0;
    }
    // rtmgr->setCache(".cachefile");
    std::shared_ptr<Module> embed(Module::load(std::vector<std::string>{}, std::vector<std::string>{}, argv[1], rtmgr));
    std::shared_ptr<Module> sam(Module::load(
        {"point_coords", "point_labels", "image_embeddings", "has_mask_input", "mask_input", "orig_im_size"},
        {"iou_predictions", "low_res_masks", "masks"}, argv[2], rtmgr));
    auto image = imread(argv[3]);
    // 1. preprocess
    auto dims = image->getInfo()->dim;
    int origin_h = dims[0];
    int origin_w = dims[1];
    int length = 1024;
    int new_h, new_w;
    if (origin_h > origin_w) {
        new_w = round(origin_w * (float)length / origin_h);
        new_h = length;
    } else {
        new_h = round(origin_h * (float)length / origin_w);
        new_w = length;
    }
    float scale_w = (float)new_w / origin_w;
    float scale_h = (float)new_h / origin_h;
    auto input_var = resize(image, Size(new_w, new_h), 0, 0, INTER_LINEAR, -1, {123.675, 116.28, 103.53}, {1/58.395, 1/57.12, 1/57.375});
    std::vector<int> padvals { 0, length - new_h, 0, length - new_w, 0, 0 };
    auto pads = _Const(static_cast<void*>(padvals.data()), {3, 2}, NCHW, halide_type_of<int>());
    input_var = _Pad(input_var, pads, CONSTANT);
    input_var = _Unsqueeze(input_var, {0});
    // 2. image embedding
    input_var = _Convert(input_var, NC4HW4);
    auto outputs = embed->onForward({input_var});
    auto image_embedding = _Convert(outputs[0], NCHW);

    // 3. segment
    auto build_input = [](std::vector<float> data, std::vector<int> shape) {
        return _Const(static_cast<void*>(data.data()), shape, NCHW, halide_type_of<float>());
    };
    // build inputs
    std::vector<float> points = {500, 375};
    auto scale_points = points;
    for (int i = 0; i < scale_points.size() / 2; i++) {
        scale_points[2 * i] = scale_points[2 * i] * scale_w;
        scale_points[2 * i + 1] = scale_points[2 * i + 1] * scale_h;
    }
    scale_points.push_back(0);
    scale_points.push_back(0);
    auto point_coords = build_input(scale_points, {1, 2, 2});
    auto point_labels = build_input({1, -1}, {1, 2});
    auto orig_im_size = build_input({static_cast<float>(origin_h), static_cast<float>(origin_w)}, {2});
    auto has_mask_input = build_input({0}, {1});
    std::vector<float> zeros(256*256, 0.f);
    auto mask_input = build_input(zeros, {1, 1, 256, 256});
    auto output_vars = sam->onForward({point_coords, point_labels, image_embedding, has_mask_input, mask_input, orig_im_size});
    auto masks = _Convert(output_vars[2], NCHW);
    // 4. postprocess: draw mask and point
    masks = _Greater(masks, _Scalar(mask_threshold));
    masks = _Reshape(masks, {origin_h, origin_w, 1});
    std::vector<int> color_vec {30, 144, 255};
    auto color = _Const(static_cast<void*>(color_vec.data()), {1, 1, 3}, NCHW, halide_type_of<int>());
    image = _Cast<uint8_t>(_Cast<int>(image) + masks * color);
    auto ptr = image->readMap<uint8_t>();
    for (int i = 0; i < points.size() / 2; i++) {
        float x = points[2 * i];
        float y = points[2 * i + 1];
        circle(image, {x, y}, 10, {0, 0, 255}, 5);
    }
    if (imwrite("res.jpg", image)) {
        MNN_PRINT("result image write to `res.jpg`.\n");
    }
    // rtmgr->updateCache();
    return 0;
}
