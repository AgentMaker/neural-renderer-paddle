import os
__dirname__ = os.path.dirname(__file__)

from paddle.utils.cpp_extension import load

src_files = [
    'create_texture_image_cuda.cc', 'create_texture_image_cuda_kernel.cu'
]
src_files = [os.path.join(__dirname__, filename) for filename in src_files]

create_texture_image_ops = load(
    name="neural_renderer_create_texture_image",
    sources=src_files)

create_texture_image = create_texture_image_ops.create_texture_image
