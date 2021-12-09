#include "paddle/extension.h"

#include <vector>

// CUDA forward declarations

std::vector<paddle::Tensor> forward_face_index_map_cuda(
    const paddle::Tensor &faces,
    const paddle::Tensor &face_index_map,
    const paddle::Tensor &weight_map,
    const paddle::Tensor &depth_map,
    const paddle::Tensor &face_inv_map,
    const paddle::Tensor &faces_inv,
    int image_size,
    float near,
    float far,
    int return_rgb,
    int return_alpha,
    int return_depth);

std::vector<paddle::Tensor> forward_texture_sampling_cuda(
    const paddle::Tensor &faces,
    const paddle::Tensor &textures,
    const paddle::Tensor &face_index_map,
    const paddle::Tensor &weight_map,
    const paddle::Tensor &depth_map,
    const paddle::Tensor &rgb_map,
    const paddle::Tensor &sampling_index_map,
    const paddle::Tensor &sampling_weight_map,
    int image_size,
    float eps);

paddle::Tensor backward_pixel_map_cuda(
    const paddle::Tensor &faces,
    const paddle::Tensor &face_index_map,
    const paddle::Tensor &rgb_map,
    const paddle::Tensor &alpha_map,
    const paddle::Tensor &grad_rgb_map,
    const paddle::Tensor &grad_alpha_map,
    const paddle::Tensor &grad_faces,
    int image_size,
    float eps,
    int return_rgb,
    int return_alpha);

paddle::Tensor backward_textures_cuda(
    const paddle::Tensor &face_index_map,
    const paddle::Tensor &sampling_weight_map,
    const paddle::Tensor &sampling_index_map,
    const paddle::Tensor &grad_rgb_map,
    const paddle::Tensor &grad_textures,
    int num_faces);

paddle::Tensor backward_depth_map_cuda(
    const paddle::Tensor &faces,
    const paddle::Tensor &depth_map,
    const paddle::Tensor &face_index_map,
    const paddle::Tensor &face_inv_map,
    const paddle::Tensor &weight_map,
    const paddle::Tensor &grad_depth_map,
    const paddle::Tensor &grad_faces,
    int image_size);

// C++ interface

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")


std::vector<paddle::Tensor> FaceIndexMapForward(
    const paddle::Tensor &faces,
    const paddle::Tensor &face_index_map,
    const paddle::Tensor &weight_map,
    const paddle::Tensor &depth_map,
    const paddle::Tensor &face_inv_map,
    const paddle::Tensor &faces_inv,
    int image_size,
    float near,
    float far,
    int return_rgb,
    int return_alpha,
    int return_depth) {

    CHECK_INPUT(faces);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(face_inv_map);
    CHECK_INPUT(faces_inv);

    return forward_face_index_map_cuda(faces, face_index_map, weight_map,
                                       depth_map, face_inv_map, faces_inv,
                                       image_size, near, far,
                                       return_rgb, return_alpha, return_depth);
}

std::vector<std::vector<int64_t>> FaceIndexMapForwardInferShape(
    std::vector<int64_t> faces_shape,
    std::vector<int64_t> face_index_map_shape,
    std::vector<int64_t> weight_map_shape,
    std::vector<int64_t> depth_map_shape,
    std::vector<int64_t> face_inv_map_shape,
    std::vector<int64_t> faces_inv_shape) {
    return {face_index_map_shape, weight_map_shape, depth_map_shape, face_inv_map_shape};
}

std::vector<paddle::DataType> FaceIndexMapForwardInferDtype(
    paddle::DataType faces_dtype,
    paddle::DataType face_index_map_dtype,
    paddle::DataType weight_map_dtype,
    paddle::DataType depth_map_dtype,
    paddle::DataType face_inv_map_dtype,
    paddle::DataType faces_inv_dtype) {
    return {face_index_map_dtype, weight_map_dtype, depth_map_dtype, face_inv_map_dtype};
}

std::vector<paddle::Tensor> TextureSamplingForward(
    const paddle::Tensor &faces,
    const paddle::Tensor &textures,
    const paddle::Tensor &face_index_map,
    const paddle::Tensor &weight_map,
    const paddle::Tensor &depth_map,
    const paddle::Tensor &rgb_map,
    const paddle::Tensor &sampling_index_map,
    const paddle::Tensor &sampling_weight_map,
    int image_size,
    float eps) {

    CHECK_INPUT(faces);
    CHECK_INPUT(textures);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(rgb_map);
    CHECK_INPUT(sampling_index_map);
    CHECK_INPUT(sampling_weight_map);

    return forward_texture_sampling_cuda(faces, textures, face_index_map,
                                    weight_map, depth_map, rgb_map,
                                    sampling_index_map, sampling_weight_map,
                                    image_size, eps);
}

std::vector<std::vector<int64_t>> TextureSamplingForwardInferShape(
    std::vector<int64_t> faces_shape,
    std::vector<int64_t> textures_shape,
    std::vector<int64_t> face_index_map_shape,
    std::vector<int64_t> weight_map_shape,
    std::vector<int64_t> depth_map_shape,
    std::vector<int64_t> rgb_map_shape,
    std::vector<int64_t> sampling_index_map_shape,
    std::vector<int64_t> sampling_weight_map_shape) {
    return {rgb_map_shape, sampling_index_map_shape, sampling_weight_map_shape};
}

std::vector<paddle::DataType> TextureSamplingForwardInferDtype(
    paddle::DataType faces_dtype,
    paddle::DataType textures_dtype,
    paddle::DataType face_index_map_dtype,
    paddle::DataType weight_map_dtype,
    paddle::DataType depth_map_dtype,
    paddle::DataType rgb_map_dtype,
    paddle::DataType sampling_index_map_dtype,
    paddle::DataType sampling_weight_map_dtype) {
    return {rgb_map_dtype, sampling_index_map_dtype, sampling_weight_map_dtype};
}

std::vector<paddle::Tensor> PixelMapBackward(
    const paddle::Tensor &faces,
    const paddle::Tensor &face_index_map,
    const paddle::Tensor &rgb_map,
    const paddle::Tensor &alpha_map,
    const paddle::Tensor &grad_rgb_map,
    const paddle::Tensor &grad_alpha_map,
    const paddle::Tensor &grad_faces,
    int image_size,
    float eps,
    int return_rgb,
    int return_alpha) {

    CHECK_INPUT(faces);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(rgb_map);
    CHECK_INPUT(alpha_map);
    CHECK_INPUT(grad_rgb_map);
    CHECK_INPUT(grad_alpha_map);
    CHECK_INPUT(grad_faces);

    return {backward_pixel_map_cuda(faces, face_index_map, rgb_map, alpha_map,
                                   grad_rgb_map, grad_alpha_map, grad_faces,
                                   image_size, eps, return_rgb, return_alpha)};
}

std::vector<std::vector<int64_t>> PixelMapBackwardInferShape(
    std::vector<int64_t> faces_shape,
    std::vector<int64_t> face_index_map_shape,
    std::vector<int64_t> rgb_map_shape,
    std::vector<int64_t> alpha_map_shape,
    std::vector<int64_t> grad_rgb_map_shape,
    std::vector<int64_t> grad_alpha_map_shape,
    std::vector<int64_t> grad_faces_shape) {
    return {grad_faces_shape};
}

std::vector<paddle::DataType> PixelMapBackwardInferDtype(
    paddle::DataType faces_dtype,
    paddle::DataType face_index_map_dtype,
    paddle::DataType rgb_map_dtype,
    paddle::DataType alpha_map_dtype,
    paddle::DataType grad_rgb_map_dtype,
    paddle::DataType grad_alpha_map_dtype,
    paddle::DataType grad_faces_dtype) {
    return {grad_faces_dtype};
}

std::vector<paddle::Tensor> TexturesBackward(
    const paddle::Tensor &face_index_map,
    const paddle::Tensor &sampling_weight_map,
    const paddle::Tensor &sampling_index_map,
    const paddle::Tensor &grad_rgb_map,
    const paddle::Tensor &grad_textures,
    int num_faces) {

    CHECK_INPUT(face_index_map);
    CHECK_INPUT(sampling_weight_map);
    CHECK_INPUT(sampling_index_map);
    CHECK_INPUT(grad_rgb_map);
    CHECK_INPUT(grad_textures);

    return {backward_textures_cuda(face_index_map, sampling_weight_map,
                                  sampling_index_map, grad_rgb_map,
                                  grad_textures, num_faces)};
}

std::vector<std::vector<int64_t>> TexturesBackwardInferShape(
    std::vector<int64_t> face_index_map_shape,
    std::vector<int64_t> sampling_weight_map_shape,
    std::vector<int64_t> sampling_index_map_shape,
    std::vector<int64_t> grad_rgb_map_shape,
    std::vector<int64_t> grad_textures_shape) {
    return {grad_textures_shape};
}

std::vector<paddle::DataType> TexturesBackwardInferDtype(
    paddle::DataType face_index_map_dtype,
    paddle::DataType sampling_weight_map_dtype,
    paddle::DataType sampling_index_map_dtype,
    paddle::DataType grad_rgb_map_dtype,
    paddle::DataType grad_textures_dtype) {
    return {grad_textures_dtype};
}

std::vector<paddle::Tensor> DepthMapBackward(
    const paddle::Tensor &faces,
    const paddle::Tensor &depth_map,
    const paddle::Tensor &face_index_map,
    const paddle::Tensor &face_inv_map,
    const paddle::Tensor &weight_map,
    const paddle::Tensor &grad_depth_map,
    const paddle::Tensor &grad_faces,
    int image_size) {

    CHECK_INPUT(faces);
    CHECK_INPUT(depth_map);
    CHECK_INPUT(face_index_map);
    CHECK_INPUT(face_inv_map);
    CHECK_INPUT(weight_map);
    CHECK_INPUT(grad_depth_map);
    CHECK_INPUT(grad_faces);

    return {backward_depth_map_cuda(faces, depth_map, face_index_map,
                                   face_inv_map, weight_map,
                                   grad_depth_map, grad_faces,
                                   image_size)};
}

std::vector<std::vector<int64_t>> DepthMapBackwardInferShape(
    std::vector<int64_t> faces_shape,
    std::vector<int64_t> depth_map_shape,
    std::vector<int64_t> face_index_map_shape,
    std::vector<int64_t> face_inv_map_shape,
    std::vector<int64_t> weight_map_shape,
    std::vector<int64_t> grad_depth_map_shape,
    std::vector<int64_t> grad_faces_shape) {
    return {grad_faces_shape};
}

std::vector<paddle::DataType> DepthMapBackwardInferDtype(
    paddle::DataType faces_dtype,
    paddle::DataType depth_map_dtype,
    paddle::DataType face_index_map_dtype,
    paddle::DataType face_inv_map_dtype,
    paddle::DataType weight_map_dtype,
    paddle::DataType grad_depth_map_dtype,
    paddle::DataType grad_faces_dtype) {
    return {grad_faces_dtype};
}

PD_BUILD_OP(forward_face_index_map)
    .Inputs({"faces", "face_index_map", "weight_map", "depth_map", "face_inv_map", "faces_inv"})
    .Outputs({"face_index_map_out", "weight_map_out", "depth_map_out", "face_inv_map_out"})
    .Attrs({
        "image_size: int",
        "near: float",
        "far: float",
        "return_rgb: int",
        "return_alpha: int",
        "return_depth: int"
    })
    .SetKernelFn(PD_KERNEL(FaceIndexMapForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FaceIndexMapForwardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FaceIndexMapForwardInferDtype));

PD_BUILD_OP(forward_texture_sampling)
    .Inputs({"faces", "textures", "face_index_map", "weight_map", "depth_map", "rgb_map", "sampling_index_map", "sampling_weight_map"})
    .Outputs({"rgb_map_out", "sampling_index_map_out", "sampling_weight_map_out"})
    .Attrs({
        "image_size: int",
        "eps: float"
    })
    .SetKernelFn(PD_KERNEL(TextureSamplingForward))
    .SetInferShapeFn(PD_INFER_SHAPE(TextureSamplingForwardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TextureSamplingForwardInferDtype));

PD_BUILD_OP(backward_pixel_map)
    .Inputs({"faces", "face_index_map", "rgb_map", "alpha_map", "grad_rgb_map", "grad_alpha_map", "grad_faces"})
    .Outputs({"grad_faces_out"})
    .Attrs({
        "image_size: int",
        "eps: float",
        "return_rgb: int",
        "return_alpha: int"
    })
    .SetKernelFn(PD_KERNEL(PixelMapBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(PixelMapBackwardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PixelMapBackwardInferDtype));

PD_BUILD_OP(backward_textures)
    .Inputs({"face_index_map", "sampling_weight_map", "sampling_index_map", "grad_rgb_map", "grad_textures"})
    .Outputs({"grad_textures_out"})
    .Attrs({
        "num_faces: int"
    })
    .SetKernelFn(PD_KERNEL(TexturesBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(TexturesBackwardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TexturesBackwardInferDtype));

PD_BUILD_OP(backward_depth_map)
    .Inputs({"faces", "depth_map", "face_index_map", "face_inv_map", "weight_map", "grad_depth_map", "grad_faces"})
    .Outputs({"grad_faces_out"})
    .Attrs({
        "image_size: int"
    })
    .SetKernelFn(PD_KERNEL(DepthMapBackward))
    .SetInferShapeFn(PD_INFER_SHAPE(DepthMapBackwardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DepthMapBackwardInferDtype));
