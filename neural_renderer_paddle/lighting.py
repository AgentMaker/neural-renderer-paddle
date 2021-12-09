import paddle
import paddle.nn.functional as F
import numpy as np

def lighting(faces, textures, intensity_ambient=0.5, intensity_directional=0.5,
             color_ambient=(1, 1, 1), color_directional=(1, 1, 1), direction=(0, 1, 0)):

    bs, nf = faces.shape[:2]
    place = faces.place

    # arguments
    # make sure to convert all inputs to float tensors
    if isinstance(color_ambient, tuple) or isinstance(color_ambient, list):
        color_ambient = paddle.to_tensor(color_ambient, dtype=paddle.float32, place=place)
    elif isinstance(color_ambient, np.ndarray):
        color_ambient = paddle.to_tensor(color_ambient, dtype=paddle.float32, place=place)
    if isinstance(color_directional, tuple) or isinstance(color_directional, list):
        color_directional = paddle.to_tensor(color_directional, dtype=paddle.float32, place=place)
    elif isinstance(color_directional, np.ndarray):
        color_directional = paddle.to_tensor(color_directional, dtype=paddle.float32, place=place)
    if isinstance(direction, tuple) or isinstance(direction, list):
        direction = paddle.to_tensor(direction, dtype=paddle.float32, place=place)
    elif isinstance(direction, np.ndarray):
        direction = paddle.to_tensor(direction, dtype=paddle.float32, place=place)
    if color_ambient.ndimension() == 1:
        color_ambient = color_ambient[None, :]
    if color_directional.ndimension() == 1:
        color_directional = color_directional[None, :]
    if direction.ndimension() == 1:
        direction = direction[None, :]

    # create light
    light = paddle.zeros([bs, nf, 3], dtype=paddle.float32).to(place)

    # ambient light
    if intensity_ambient != 0:
        light += intensity_ambient * color_ambient[:, None, :]

    # directional light
    if intensity_directional != 0:
        faces = faces.reshape((bs * nf, 3, 3))
        v10 = faces[:, 0] - faces[:, 1]
        v12 = faces[:, 2] - faces[:, 1]
        # paddle normalize divides by max(norm, eps) instead of (norm+eps) in chainer
        normals = F.normalize(paddle.cross(v10, v12), epsilon=1e-5)
        normals = normals.reshape((bs, nf, 3))

        if direction.ndimension() == 2:
            direction = direction[:, None, :]
        cos = F.relu(paddle.sum(normals * direction, axis=2))
        # may have to verify that the next line is correct
        light += intensity_directional * (color_directional[:, None, :] * cos[:, :, None])

    # apply
    light = light[:,:,None, None, None, :]
    textures *= light
    return textures
