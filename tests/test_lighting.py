import unittest

import paddle

import neural_renderer_paddle as nr

class TestLighting(unittest.TestCase):
    
    def test_case1(self):
        """Test whether it is executable."""
        faces = paddle.randn([64, 16, 3, 3], dtype=paddle.float32)
        textures = paddle.randn([64, 16, 8, 8, 8, 3], dtype=paddle.float32)
        nr.lighting(faces, textures)

if __name__ == '__main__':
    unittest.main()



