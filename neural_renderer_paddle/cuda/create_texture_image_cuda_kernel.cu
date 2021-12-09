#include <paddle/extension.h>

#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template<typename data_t>
__global__ void create_texture_image_cuda_kernel(
    const data_t* __restrict__ vertices_all,
    const data_t* __restrict__ textures,
    data_t* __restrict__ image,
    size_t image_size,
    size_t num_faces,
    size_t texture_size_in,
    size_t texture_size_out,
    size_t tile_width,
    data_t eps) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= image_size / 3) {
        return;
    }
    const int x = i % (tile_width * texture_size_out);
    const int y = i / (tile_width * texture_size_out);
    const int row = x / texture_size_out;
    const int column = y / texture_size_out;
    const int fn = row + column * tile_width;
    const int tsi = texture_size_in;

    const data_t* texture = &textures[fn * tsi * tsi * tsi * 3];
    const data_t* vertices = &vertices_all[fn * 3 * 2];
    const data_t* p0 = &vertices[2 * 0];
    const data_t* p1 = &vertices[2 * 1];
    const data_t* p2 = &vertices[2 * 2];

    /* */
    // if ((y % ${texture_size_out}) < (x % ${texture_size_out})) continue;

    /* compute face_inv */
    data_t face_inv[9] = {
        p1[1] - p2[1], p2[0] - p1[0], p1[0] * p2[1] - p2[0] * p1[1],
        p2[1] - p0[1], p0[0] - p2[0], p2[0] * p0[1] - p0[0] * p2[1],
        p0[1] - p1[1], p1[0] - p0[0], p0[0] * p1[1] - p1[0] * p0[1]};
    data_t face_inv_denominator = (
        p2[0] * (p0[1] - p1[1]) +
        p0[0] * (p1[1] - p2[1]) +
        p1[0] * (p2[1] - p0[1]));
    for (int k = 0; k < 9; k++) face_inv[k] /= face_inv_denominator;

    /* compute w = face_inv * p */
    data_t weight[3];
    data_t weight_sum = 0;
    for (int k = 0; k < 3; k++) {
        weight[k] = face_inv[3 * k + 0] * x + face_inv[3 * k + 1] * y + face_inv[3 * k + 2];
        weight_sum += weight[k];
    }
    for (int k = 0; k < 3; k++)
        weight[k] /= (weight_sum + eps);

    /* get texture index (data_t) */
    data_t texture_index_data_t[3];
    for (int k = 0; k < 3; k++) {
        data_t tif = weight[k] * (tsi - 1);
        tif = max(tif, 0.);
        tif = min(tif, tsi - 1 - eps);
        texture_index_data_t[k] = tif;
    }

    /* blend */
    data_t new_pixel[3] = {0, 0, 0};
    for (int pn = 0; pn < 8; pn++) {
        data_t w = 1;                         // weight
        int texture_index_int[3];            // index in source (int)
        for (int k = 0; k < 3; k++) {
            if ((pn >> k) % 2 == 0) {
                w *= 1 - (texture_index_data_t[k] - (int)texture_index_data_t[k]);
                texture_index_int[k] = (int)texture_index_data_t[k];
            }
            else {
                w *= texture_index_data_t[k] - (int)texture_index_data_t[k];
                texture_index_int[k] = (int)texture_index_data_t[k] + 1;
            }
        }
        int isc = texture_index_int[0] * tsi * tsi + texture_index_int[1] * tsi + texture_index_int[2];
        for (int k = 0; k < 3; k++)
            new_pixel[k] += w * texture[isc * 3 + k];
    }
    for (int k = 0; k < 3; k++)
        image[i * 3 + k] = new_pixel[k];
}

// didn't really look to see if we fuse the 2 kernels
// probably not because of synchronization issues
template<typename data_t>
__global__ void create_texture_image_boundary_cuda_kernel(
        data_t* image,
        size_t image_size,
        size_t texture_size_out,
        size_t tile_width) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= image_size / 3) {
        return;
    }

    const int x = i % (tile_width * texture_size_out);
    const int y = i / (tile_width * texture_size_out);
    if ((y % texture_size_out + 1) == (x % texture_size_out)) {
      for (int k = 0; k < 3; k++)
          image[i * 3 + k] = 
              image[ (y * tile_width * texture_size_out + (x - 1))  * 3 + k];
    }
}

}

paddle::Tensor create_texture_image_cuda(
    const paddle::Tensor &vertices_all,
    const paddle::Tensor &textures,
    const paddle::Tensor &image,
    float eps) {

    const auto num_faces = textures.shape()[0];
    const auto texture_size_in = textures.shape()[1];
    const auto tile_width = int(sqrt(num_faces - 1)) + 1;
    const auto texture_size_out = image.shape()[1] / tile_width;

    const int threads = 128;
    const int image_size = image.size();
    const dim3 blocks ((image_size / 3 - 1) / threads + 1, 1, 1);

    PD_DISPATCH_FLOATING_TYPES(image.type(), "create_texture_image_cuda", ([&] {
        create_texture_image_cuda_kernel<data_t><<<blocks, threads>>>(
            vertices_all.data<data_t>(),
            textures.data<data_t>(),
            const_cast<data_t*>(image.data<data_t>()),
            image_size,
            num_faces,
            texture_size_in,
            texture_size_out,
            tile_width,
            (data_t) eps);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        PD_THROW("Error in create_texture_image: %s\n", cudaGetErrorString(err));

    PD_DISPATCH_FLOATING_TYPES(image.type(), "create_texture_image_boundary", ([&] {
        create_texture_image_boundary_cuda_kernel<data_t><<<blocks, threads>>>(
            const_cast<data_t*>(image.data<data_t>()),
            image_size,
            texture_size_out,
            tile_width);
    }));

    err = cudaGetLastError();
    if (err != cudaSuccess) 
        PD_THROW("Error in create_texture_image_boundary: %s\n", cudaGetErrorString(err));

    return image;
}
