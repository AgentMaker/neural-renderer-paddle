import numpy as np
import paddle
import paddle.nn.functional as F


def look(vertices, eye, direction=[0, 1, 0], up=None):
    """
    "Look" transformation of vertices.
    """
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')

    place = vertices.place

    if isinstance(direction, list) or isinstance(direction, tuple):
        direction = paddle.to_tensor(direction, dtype=paddle.float32, place=place)
    elif isinstance(direction, np.ndarray):
        direction = paddle.to_tensor(direction).to(place)
    elif paddle.is_tensor(direction):
        direction = direction.to(place)

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = paddle.to_tensor(eye, dtype=paddle.float32, place=place)
    elif isinstance(eye, np.ndarray):
        eye = paddle.to_tensor(eye).to(place)
    elif paddle.is_tensor(eye):
        eye = eye.to(place)

    if up is None:
        up = paddle.to_tensor([0, 1, 0])
    if eye.ndimension() == 1:
        eye = eye[None, :]
    if direction.ndimension() == 1:
        direction = direction[None, :]
    if up.ndimension() == 1:
        up = up[None, :]

    # prevent paddle no grad error
    direction.stop_gradient = False
    eye.stop_gradient = False
    up.stop_gradient = False

    # create new axes
    z_axis = F.normalize(direction, epsilon=1e-5)
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
