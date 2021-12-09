import os
__dirname__ = os.path.dirname(__file__)

from paddle.utils.cpp_extension import load

src_files = [
    'load_textures_cuda.cc', 'load_textures_cuda_kernel.cu'
]
src_files = [os.path.join(__dirname__, filename) for filename in src_files]

load_textures_ops = load(
    name="neural_renderer_load_textures",
    sources=src_files)

load_textures = load_textures_ops.load_textures
