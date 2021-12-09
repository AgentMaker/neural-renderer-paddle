import os
__dirname__ = os.path.dirname(__file__)

from paddle.utils.cpp_extension import load

src_files = [
    'rasterize_cuda.cc', 'rasterize_cuda_kernel.cu'
]
src_files = [os.path.join(__dirname__, filename) for filename in src_files]

rasterize_ops = load(
    name="neural_renderer_rasterize",
    sources=src_files)

forward_face_index_map = rasterize_ops.forward_face_index_map
forward_texture_sampling = rasterize_ops.forward_texture_sampling
backward_pixel_map = rasterize_ops.backward_pixel_map
backward_textures = rasterize_ops.backward_textures
backward_depth_map = rasterize_ops.backward_depth_map
