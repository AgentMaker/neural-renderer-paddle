import os

import paddle

import neural_renderer_paddle as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def to_minibatch(data, batch_size=4, target_num=2):
    ret = []
    for d in data:
        place = d.place
        d2 = paddle.unsqueeze(paddle.zeros_like(d), 0)
        r = [1 for _ in d2.shape]
        r[0] = batch_size
        d2 = paddle.unsqueeze(paddle.zeros_like(d), 0).tile(r).to(place)
        d2[target_num] = d
        ret.append(d2)
    return ret

def load_teapot_batch(batch_size=4, target_num=2):
    vertices, faces = nr.load_obj(os.path.join(data_dir, 'teapot.obj'))
    textures = paddle.ones([faces.shape[0], 4, 4, 4, 3], dtype=paddle.float32)
    vertices, faces, textures = to_minibatch((vertices, faces, textures), batch_size, target_num)
    return vertices, faces, textures
