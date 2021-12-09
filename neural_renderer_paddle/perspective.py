from __future__ import division
import math

import paddle

def perspective(vertices, angle=30.):
    '''
    Compute perspective distortion from a given angle
    '''
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')
    place = vertices.place
    angle = paddle.to_tensor(angle / 180 * math.pi, dtype=paddle.float32, place=place) # shape=[1] in paddle
    width = paddle.tan(angle)
    width = width[:, None] 
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = paddle.stack((x,y,z), axis=2)
    return vertices
