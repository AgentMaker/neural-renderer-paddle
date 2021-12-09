import paddle


def vertices_to_faces(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3)
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    place = vertices.place
    faces = faces + (paddle.arange(bs, dtype=paddle.int32).to(place) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # paddle only supports long and byte tensors for indexing
    return vertices[faces.astype(paddle.int64)]
