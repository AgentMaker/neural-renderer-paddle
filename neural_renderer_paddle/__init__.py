from .get_points_from_angles import get_points_from_angles
from .lighting import lighting
from .load_obj import load_obj
from .look import look
from .look_at import look_at
from .mesh import Mesh
from .perspective import perspective
from .projection import projection
from .rasterize import (rasterize_rgbad, rasterize, rasterize_silhouettes, rasterize_depth, Rasterize)
from .renderer import Renderer
from .save_obj import save_obj
from .vertices_to_faces import vertices_to_faces

__version__ = '1.1.3'
name = 'neural_renderer_paddle'

# add extra ops for paddle

import paddle


set_device = paddle.set_device
paddle.set_device = lambda device: set_device(device.replace('cuda', 'gpu'))

def to(self, place, dtype=None):
    if isinstance(place, str):
        if place == 'cpu':
            place = paddle.CPUPlace()
        elif place == 'cuda':
            place = paddle.CUDAPlace(0)
        elif 'cuda:' in place:
            place =  paddle.CUDAPlace(int(place.split(':')[1]))
    out = self
    if isinstance(dtype, str):
        dtype = getattr(paddle, dtype)
    if dtype is not None and self.dtype != dtype:
        out = self.astype(dtype)
    if self.place._equals(place):
        return out
    out = paddle.to_tensor(out, place=place, stop_gradient=self.stop_gradient)
    if self.grad is not None:
        grad = self.grad.to(place, dtype)
        out._set_grad_ivar(grad)
    return out

paddle.Tensor.to = to
paddle.Tensor.cpu = lambda self: to(self, 'cpu')
paddle.Tensor.cuda = lambda self: to(self, 'cuda')


_layer_to = paddle.nn.Layer.to
def layer_to(self, device=None, dtype=None, blocking=None):
    if isinstance(device, str):
        if device == 'cpu':
            place = paddle.CPUPlace()
        elif device == 'cuda':
            place = paddle.CUDAPlace(0)
        elif 'cuda:' in device:
            place =  paddle.CUDAPlace(int(device.split(':')[1]))
    if isinstance(dtype, paddle.dtype):
        dtype = str(dtype).split('.')[-1]
    if self.parameters()[0].place._equals(place):
        device = None
    if self.parameters()[0].dtype == dtype:
        dtype = None
    if device is None and dtype is None:
        return self
    _layer_to(self, device, dtype, blocking)
    return self

paddle.nn.Layer.to = layer_to


def swapaxes(self, a, b):
    dims = list(range(self.ndim))
    dims[a], dims[b] = dims[b], dims[a]

    return self.transpose(dims)

paddle.Tensor.swapaxes = swapaxes
