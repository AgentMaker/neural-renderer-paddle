import numpy as np
import paddle
import paddle.nn.functional as F


def look_at(vertices, eye, at=[0, 0, 0], up=[0, 1, 0]):
    """
    "Look at" transformation of vertices.
    """
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')

    place = vertices.place

    # if list or tuple convert to numpy array
    if isinstance(at, list) or isinstance(at, tuple):
        at = paddle.to_tensor(at, dtype=paddle.float32, place=place)
    # if numpy array convert to tensor
    elif isinstance(at, np.ndarray):
        at = paddle.to_tensor(at).to(place)
    elif paddle.is_tensor(at):
        at = at.to(place)

    if isinstance(up, list) or isinstance(up, tuple):
        up = paddle.to_tensor(up, dtype=paddle.float32, place=place)
    elif isinstance(up, np.ndarray):
        up = paddle.to_tensor(up).to(place)
    elif paddle.is_tensor(up):
        up = up.to(place)

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = paddle.to_tensor(eye, dtype=paddle.float32, place=place)
    elif isinstance(eye, np.ndarray):
        eye = paddle.to_tensor(eye).to(place)
    elif paddle.is_tensor(eye):
        eye = eye.to(place)

    batch_size = vertices.shape[0]
    if eye.ndimension() == 1:
        eye = eye[None, :].tile([batch_size, 1])
    if at.ndimension() == 1:
        at = at[None, :].tile([batch_size, 1])
    if up.ndimension() == 1:
        up = up[None, :].tile([batch_size, 1])
    
    # prevent paddle no grad error
    at.stop_gradient = False
    eye.stop_gradient = False
    up.stop_gradient = False

    # create new axes
    # eps is chosen as 0.5 to match the chainer version
    z_axis = F.normalize(at - eye, epsilon=1e-5)
    x_axis = F.normalize(paddle.cross(up, z_axis), epsilon=1e-5)
    y_axis = F.normalize(paddle.cross(z_axis, x_axis), epsilon=1e-5)

    # create rotation matrix: [bs, 3, 3]
    r = paddle.concat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), axis=1)

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = eye[:, None, :]
    vertices = vertices - eye
    vertices = paddle.matmul(vertices, r.swapaxes(1,2))

    return vertices
