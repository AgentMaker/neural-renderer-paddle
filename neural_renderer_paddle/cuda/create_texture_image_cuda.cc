#include "paddle/extension.h"

#include <vector>

// CUDA forward declarations

paddle::Tensor create_texture_image_cuda(
        const paddle::Tensor &vertices_all,
        const paddle::Tensor &textures,
        const paddle::Tensor &image,
        float eps);

// C++ interface

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")


std::vector<paddle::Tensor> CreateTextureImage(
        const paddle::Tensor &vertices_all,
        const paddle::Tensor &textures,
        const paddle::Tensor &image,
        float eps) {

    CHECK_INPUT(vertices_all);
    CHECK_INPUT(textures);
    CHECK_INPUT(image);
    
    return {create_texture_image_cuda(vertices_all, textures, image, eps)};
}

std::vector<std::vector<int64_t>> CreateTextureImageInferShape(
    std::vector<int64_t> vertices_all_shape,
    std::vector<int64_t> textures_shape,
    std::vector<int64_t> image_shape) {
    return {image_shape};
}

std::vector<paddle::DataType> CreateTextureImageInferDtype(
    paddle::DataType vertices_all_dtype,
    paddle::DataType textures_dtype,
    paddle::DataType image_dtype) {
    return {image_dtype};
}

PD_BUILD_OP(create_texture_image)
    .Inputs({"vertices_all", "textures", "image"})
    .Outputs({"image_out"})
    .Attrs({"eps: float"})
    .SetKernelFn(PD_KERNEL(CreateTextureImage))
    .SetInferShapeFn(PD_INFER_SHAPE(CreateTextureImageInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(CreateTextureImageInferDtype));
