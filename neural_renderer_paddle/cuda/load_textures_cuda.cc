#include "paddle/extension.h"

#include <vector>

// CUDA forward declarations

paddle::Tensor load_textures_cuda(
    const paddle::Tensor &image,
    const paddle::Tensor &faces,
    const paddle::Tensor &textures,
    const paddle::Tensor &is_update,
    int texture_wrapping,
    int use_bilinear);

// C++ interface

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")


std::vector<paddle::Tensor> LoadTextures(
    const paddle::Tensor &image,
    const paddle::Tensor &faces,
    const paddle::Tensor &textures,
    const paddle::Tensor &is_update,
    int texture_wrapping,
    int use_bilinear) {

    CHECK_INPUT(image);
    CHECK_INPUT(faces);
    CHECK_INPUT(is_update);
    CHECK_INPUT(textures);

    return {load_textures_cuda(image, faces, textures, is_update, texture_wrapping, use_bilinear)};
                                      
}

std::vector<std::vector<int64_t>> LoadTexturesInferShape(
    std::vector<int64_t> image_shape,
    std::vector<int64_t> faces_shape,
    std::vector<int64_t> textures_shape,
    std::vector<int64_t> is_update_shape) {
    return {textures_shape};
}

std::vector<paddle::DataType> LoadTexturesInferDtype(
    paddle::DataType image_dtype,
    paddle::DataType faces_dtype,
    paddle::DataType textures_dtype,
    paddle::DataType is_update_dtype) {
    return {textures_dtype};
}

PD_BUILD_OP(load_textures)
    .Inputs({"image", "faces", "textures", "is_update"})
    .Outputs({"textures_out"})
    .Attrs({"texture_wrapping: int", "use_bilinear: int"})
    .SetKernelFn(PD_KERNEL(LoadTextures))
    .SetInferShapeFn(PD_INFER_SHAPE(LoadTexturesInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(LoadTexturesInferDtype));
