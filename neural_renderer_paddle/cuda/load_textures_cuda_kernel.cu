#include <paddle/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename data_t>
static __inline__ __device__ data_t mod(data_t x, data_t y) {
    if (x > 0) {
        return fmod(x,y);
    }
    else {
        return y + fmod(x,y);
    }
}

namespace {

const int REPEAT = 0;
const int MIRRORED_REPEAT = 1;
const int CLAMP_TO_EDGE = 2;
const int CLAMP_TO_BORDER = 3;

template <typename data_t>
__global__ void load_textures_cuda_kernel(
    const data_t* image,
    const int32_t* is_update,
    data_t* faces,
    data_t* __restrict__ textures, 
    int textures_size,
    int texture_size,
    int image_height,
    int image_width,
    int texture_wrapping,
    bool use_bilinear) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= textures_size / 3) {
        return;
    }
    const int ts = texture_size;
    const int fn = i / (ts * ts * ts);
    data_t dim0 = ((i / (ts * ts)) % ts) / (ts - 1.) ;
    data_t dim1 = ((i / ts) % ts) / (ts - 1.);
    data_t dim2 = (i % ts) / (ts - 1.);
    if (0 < dim0 + dim1 + dim2) {
        float sum = dim0 + dim1 + dim2;
        dim0 /= sum;
        dim1 /= sum;
        dim2 /= sum;
    }
    data_t* face = &faces[fn * 3 * 2];
    data_t* texture_ = &textures[i * 3];

    if (is_update[fn] != 0) {
        if (texture_wrapping == REPEAT) {
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                face[i] = mod(face[i], (data_t)1.);
            }
        }
        else if (texture_wrapping == MIRRORED_REPEAT) {
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                if (mod(face[i], (data_t)2) < 1) {
                    face[i] = mod(face[i], (data_t)1.);
                }
                else {
                    face[i] = 1 - mod(face[i], (data_t)1.);
                }
            }
        }
        else if (texture_wrapping == CLAMP_TO_EDGE) {
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                face[i] = max(min(face[i], (data_t) 1), (data_t) 0);
            }
        }
        const data_t pos_x = (
            (face[2 * 0 + 0] * dim0 + face[2 * 1 + 0] * dim1 + face[2 * 2 + 0] * dim2) * (image_width - 1));
        const data_t pos_y = (
            (face[2 * 0 + 1] * dim0 + face[2 * 1 + 1] * dim1 + face[2 * 2 + 1] * dim2) * (image_height - 1));
        if (use_bilinear) {
            /* bilinear sampling */
            const data_t weight_x1 = pos_x - (int)pos_x;
            const data_t weight_x0 = 1 - weight_x1;
            const data_t weight_y1 = pos_y - (int)pos_y;
            const data_t weight_y0 = 1 - weight_y1;
            for (int k = 0; k < 3; k++) {
                if (texture_wrapping != CLAMP_TO_BORDER) {
                    data_t c = 0;
                    c += image[(int)pos_y * image_width * 3 + (int)pos_x * 3 + k] * (weight_x0 * weight_y0);
                    c += image[min((int)(pos_y + 1), image_height-1) * image_width * 3 + (int)pos_x * 3 + k] * (weight_x0 * weight_y1);
                    c += image[(int)pos_y * image_width * 3 + min((int)pos_x + 1, image_width-1) * 3 + k] * (weight_x1 * weight_y0);
                    c += image[min((int)(pos_y + 1), image_height-1) * image_width * 3 + min((int)pos_x + 1, image_width-1) * 3 + k] * (weight_x1 * weight_y1);
                    texture_[k] = c;
                }
                else {
                    texture_[k] = 0;
                }
            }
        } else {
            /* nearest neighbor */
            const int pos_xi = round(pos_x);
            const int pos_yi = round(pos_y);
            for (int k = 0; k < 3; k++) {
                if (texture_wrapping != CLAMP_TO_BORDER) {
                    texture_[k] = image[pos_yi * image_width * 3 + pos_xi * 3 + k];
                }
                else {
                    texture_[k] = 0;
                }
            }
        }
    }
}

}

paddle::Tensor load_textures_cuda(
    const paddle::Tensor &image,
    const paddle::Tensor &faces,
    const paddle::Tensor &textures,
    const paddle::Tensor &is_update,
    int texture_wrapping,
    int use_bilinear) {
    // textures_size = size of the textures tensor
    const auto textures_size = textures.size();
    // notice that texture_size != texture_size
    const auto texture_size = textures.shape()[1];
    const auto image_height = image.shape()[0];
    const auto image_width = image.shape()[1];
    
    const int threads = 1024;
    const dim3 blocks ((textures_size / 3 - 1) / threads + 1);

    PD_DISPATCH_FLOATING_TYPES(image.type(), "load_textures_cuda", ([&] {
        load_textures_cuda_kernel<data_t><<<blocks, threads>>>(
            image.data<data_t>(),
            is_update.data<int32_t>(),
            faces.data<data_t>(),
            const_cast<data_t*>(textures.data<data_t>()),
            textures_size,
            texture_size,
            image_height,
            image_width,
            texture_wrapping,
            use_bilinear);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        PD_THROW("Error in load_textures: %s\n", cudaGetErrorString(err));
    return textures;
}
