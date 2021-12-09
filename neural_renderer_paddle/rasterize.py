import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer

import neural_renderer_paddle.cuda.rasterize as rasterize_cuda

DEFAULT_IMAGE_SIZE = 256
DEFAULT_ANTI_ALIASING = True
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 100
DEFAULT_EPS = 1e-4
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)

class RasterizeFunction(PyLayer):
    '''
    Definition of differentiable rasterize operation
    Some parts of the code are implemented in CUDA
    Currently implemented only for cuda Tensors
    '''
    @staticmethod
    def forward(ctx, faces, textures, image_size, near, far, eps, background_color,
                return_rgb=False, return_alpha=False, return_depth=False):
        '''
        Forward pass
        '''
        ctx.image_size = image_size
        ctx.near = near
        ctx.far = far
        ctx.eps = eps
        ctx.background_color = background_color
        ctx.return_rgb = return_rgb
        ctx.return_alpha = return_alpha
        ctx.return_depth = return_depth

        faces = faces.clone()

        ctx.place = faces.place
        ctx.batch_size, ctx.num_faces = faces.shape[:2]

        ctx.texture_stop_gradient = True
        if textures is not None:
            ctx.texture_stop_gradient = textures.stop_gradient

        if ctx.return_rgb:
            textures = textures.clone()
            ctx.texture_size = textures.shape[2]
        else:
            # initializing with dummy values
            textures = paddle.full([1], 0.0)
            ctx.texture_size = None


        face_index_map = paddle.full([ctx.batch_size, ctx.image_size, ctx.image_size], -1, paddle.int32)
        weight_map = paddle.full([ctx.batch_size, ctx.image_size, ctx.image_size, 3], 0.0)
        depth_map = paddle.full([ctx.batch_size, ctx.image_size, ctx.image_size], ctx.far)

        if ctx.return_rgb:
            rgb_map = paddle.full([ctx.batch_size, ctx.image_size, ctx.image_size, 3], 0.0)
            sampling_index_map = paddle.full([ctx.batch_size, ctx.image_size, ctx.image_size, 8], 0.0, paddle.int32)
            sampling_weight_map = paddle.full([ctx.batch_size, ctx.image_size, ctx.image_size, 8], 0.0)
        else:
            rgb_map = paddle.full([1], 0.)
            sampling_index_map = paddle.full([1], 0.)
            sampling_weight_map = paddle.full([1], 0.)
        if ctx.return_alpha:
            alpha_map = paddle.full([ctx.batch_size, ctx.image_size, ctx.image_size], 0.)
        else:
            alpha_map = paddle.full([1], 0.)
        if ctx.return_depth:
            face_inv_map = paddle.full([ctx.batch_size, ctx.image_size, ctx.image_size, 3, 3], 0.)
        else:
            face_inv_map = paddle.full([1], 0.)

        face_index_map, weight_map, depth_map, face_inv_map =\
            RasterizeFunction.forward_face_index_map(ctx, faces, face_index_map,
                                                     weight_map, depth_map,
                                                     face_inv_map)

        rgb_map, sampling_index_map, sampling_weight_map =\
                RasterizeFunction.forward_texture_sampling(ctx, faces, textures,
                                                           face_index_map, weight_map,
                                                           depth_map, rgb_map,
                                                           sampling_index_map,
                                                           sampling_weight_map)
                
        rgb_map = RasterizeFunction.forward_background(ctx, face_index_map, rgb_map)
        alpha_map = RasterizeFunction.forward_alpha_map(ctx, alpha_map, face_index_map)

        ctx.save_for_backward(faces, textures, face_index_map, weight_map,
                              depth_map, rgb_map, alpha_map, face_inv_map,
                              sampling_index_map, sampling_weight_map)


        rgb_r, alpha_r, depth_r = paddle.to_tensor([]), paddle.to_tensor([]), paddle.to_tensor([])
        if ctx.return_rgb:
            rgb_r = rgb_map
        if ctx.return_alpha:
            alpha_r = alpha_map.clone()
        if ctx.return_depth:
            depth_r = depth_map.clone()
        return rgb_r, alpha_r, depth_r

    @staticmethod
    def backward(ctx, grad_rgb_map, grad_alpha_map, grad_depth_map):
        '''
        Backward pass
        '''
        faces, textures, face_index_map, weight_map,\
        depth_map, rgb_map, alpha_map, face_inv_map,\
        sampling_index_map, sampling_weight_map = \
                ctx.saved_tensor()
        # initialize output buffers
        # no need for explicit allocation of paddle.full because zeros_like does it automatically
        grad_faces = paddle.zeros_like(faces, dtype=paddle.float32)
        if ctx.return_rgb:
            grad_textures = paddle.zeros_like(textures, dtype=paddle.float32)
        else:
            grad_textures = paddle.full([1], 0.)
        
        # get grad_outputs
        if ctx.return_rgb:
            if grad_rgb_map is not None:
                grad_rgb_map = grad_rgb_map.clone()
            else:
                grad_rgb_map = paddle.zeros_like(rgb_map)
        else:
            grad_rgb_map = paddle.full([1], 0.)
        if ctx.return_alpha:
            if grad_alpha_map is not None:
                grad_alpha_map = grad_alpha_map.clone()
            else:
                grad_alpha_map = paddle.zeros_like(alpha_map)
        else:
            grad_alpha_map = paddle.full([1], 0.)
        if ctx.return_depth:
            if grad_depth_map is not None:
                grad_depth_map = grad_depth_map.clone()
            else:
                grad_depth_map = paddle.zeros_like(ctx.depth_map)
        else:
            grad_depth_map = paddle.full([1], 0.)

        # backward pass
        grad_faces = RasterizeFunction.backward_pixel_map(
                                        ctx, faces, face_index_map, rgb_map,
                                        alpha_map, grad_rgb_map, grad_alpha_map,
                                        grad_faces)
        grad_textures = RasterizeFunction.backward_textures(
                                        ctx, face_index_map, sampling_weight_map,
                                        sampling_index_map, grad_rgb_map, grad_textures)
        grad_faces = RasterizeFunction.backward_depth_map(
                                        ctx, faces, depth_map, face_index_map,
                                        face_inv_map, weight_map, grad_depth_map,
                                        grad_faces)

        if ctx.texture_stop_gradient:
            grad_textures = None

        if ctx.texture_size is None:
            return grad_faces

        # return grad_faces, grad_textures, None, None, None, None, None, None, None, None
        # paddle PyLayer backward need num(forward input tensors) == num(backward output tensors)
        return grad_faces, grad_textures

    @staticmethod
    def forward_face_index_map(ctx, faces, face_index_map, weight_map, 
                               depth_map, face_inv_map):
        faces_inv = paddle.zeros_like(faces)
        return rasterize_cuda.forward_face_index_map(faces, face_index_map, weight_map,
                                        depth_map, face_inv_map, faces_inv,
                                        ctx.image_size, ctx.near, ctx.far,
                                        ctx.return_rgb, ctx.return_alpha,
                                        ctx.return_depth)

    @staticmethod
    def forward_texture_sampling(ctx, faces, textures, face_index_map,
                                 weight_map, depth_map, rgb_map,
                                 sampling_index_map, sampling_weight_map):
        if not ctx.return_rgb:
            return rgb_map, sampling_index_map, sampling_weight_map
        else:
            return rasterize_cuda.forward_texture_sampling(faces, textures, face_index_map,
                                           weight_map, depth_map, rgb_map,
                                           sampling_index_map, sampling_weight_map,
                                           ctx.image_size, ctx.eps)

    @staticmethod
    def forward_alpha_map(ctx, alpha_map, face_index_map):
        if ctx.return_alpha:
            alpha_map[face_index_map >= 0] = 1
        return alpha_map

    @staticmethod
    def forward_background(ctx, face_index_map, rgb_map):
        if ctx.return_rgb:
            background_color = paddle.to_tensor(ctx.background_color, dtype=paddle.float32)
            mask = (face_index_map >= 0).astype(paddle.float32)[:, :, :, None]
            if background_color.ndimension() == 1:
                rgb_map = rgb_map * mask + (1-mask) * background_color[None, None, None, :]
            elif background_color.ndimension() == 2:
                rgb_map = rgb_map * mask + (1-mask) * background_color[:, None, None, :]
        return rgb_map

    @staticmethod
    def backward_pixel_map(ctx, faces, face_index_map, rgb_map,
                           alpha_map, grad_rgb_map, grad_alpha_map, grad_faces):
        if (not ctx.return_rgb) and (not ctx.return_alpha):
            return grad_faces
        else:
            return rasterize_cuda.backward_pixel_map(faces, face_index_map, rgb_map,
                                     alpha_map, grad_rgb_map, grad_alpha_map,
                                     grad_faces, ctx.image_size, ctx.eps, ctx.return_rgb,
                                     ctx.return_alpha)

    @staticmethod
    def backward_textures(ctx, face_index_map, sampling_weight_map,
                          sampling_index_map, grad_rgb_map, grad_textures):
        if not ctx.return_rgb:
            return grad_textures
        else:
            return rasterize_cuda.backward_textures(face_index_map, sampling_weight_map,
                                                    sampling_index_map, grad_rgb_map,
                                                    grad_textures, ctx.num_faces)

    @staticmethod
    def backward_depth_map(ctx, faces, depth_map, face_index_map,
                           face_inv_map, weight_map, grad_depth_map, grad_faces):
        if not ctx.return_depth:
            return grad_faces
        else:
            return rasterize_cuda.backward_depth_map(faces, depth_map, face_index_map,
                                     face_inv_map, weight_map,
                                     grad_depth_map, grad_faces, ctx.image_size)

class Rasterize(nn.Layer):
    '''
    Wrapper around the autograd function RasterizeFunction
    Currently implemented only for cuda Tensors
    '''
    def __init__(self, image_size, near, far, eps, background_color,
                 return_rgb=False, return_alpha=False, return_depth=False):
        super(Rasterize, self).__init__()
        self.image_size = image_size
        self.image_size = image_size
        self.near = near
        self.far = far
        self.eps = eps
        self.background_color = background_color
        self.return_rgb = return_rgb
        self.return_alpha = return_alpha
        self.return_depth = return_depth

    def forward(self, faces, textures):
        if faces.place._equals(paddle.CPUPlace()) or (textures is not None and textures.place._equals(paddle.CPUPlace())):
            raise TypeError('Rasterize module supports only cuda Tensors')
        return RasterizeFunction.apply(faces, textures, self.image_size, self.near, self.far,
                                       self.eps, self.background_color,
                                       self.return_rgb, self.return_alpha, self.return_depth)

def rasterize_rgbad(
        faces,
        textures=None,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        background_color=DEFAULT_BACKGROUND_COLOR,
        return_rgb=True,
        return_alpha=True,
        return_depth=True,
):
    """
    Generate RGB, alpha channel, and depth images from faces and textures (for RGB).

    Args:
        faces (paddle.Tensor): Faces. The shape is [batch size, number of faces, 3 (vertices), 3 (XYZ)].
        textures (paddle.Tensor): Textures.
            The shape is [batch size, number of faces, texture size, texture size, texture size, 3 (RGB)].
        image_size (int): Width and height of rendered images.
        anti_aliasing (bool): do anti-aliasing by super-sampling.
        near (float): nearest z-coordinate to draw.
        far (float): farthest z-coordinate to draw.
        eps (float): small epsilon for approximated differentiation.
        background_color (tuple): background color of RGB images.
        return_rgb (bool): generate RGB images or not.
        return_alpha (bool): generate alpha channels or not.
        return_depth (bool): generate depth images or not.

    Returns:
        dict:
            {
                'rgb': RGB images. The shape is [batch size, 3, image_size, image_size].
                'alpha': Alpha channels. The shape is [batch size, image_size, image_size].
                'depth': Depth images. The shape is [batch size, image_size, image_size].
            }

    """
    if textures is None:
        inputs = [faces, None]
    else:
        inputs = [faces, textures]

    if anti_aliasing:
        # 2x super-sampling
        rgb, alpha, depth = Rasterize(
            image_size * 2, near, far, eps, background_color, return_rgb, return_alpha, return_depth)(*inputs)
    else:
        rgb, alpha, depth = Rasterize(
            image_size, near, far, eps, background_color, return_rgb, return_alpha, return_depth)(*inputs)

    # transpose & vertical flip
    if return_rgb:
        rgb = rgb.transpose((0, 3, 1, 2))
        rgb = rgb.flip([2])
    if return_alpha:
        alpha = alpha.flip([1])
    if return_depth:
        depth = depth.flip([1])

    if anti_aliasing:
        # 0.5x down-sampling
        if return_rgb:
            rgb = F.avg_pool2d(rgb, kernel_size=(2,2))
        if return_alpha:
            alpha = F.avg_pool2d(alpha[:, None, :, :], kernel_size=(2, 2))[:, 0]
        if return_depth:
            depth = F.avg_pool2d(depth[:, None, :, :], kernel_size=(2, 2))[:, 0]

    ret = {
        'rgb': rgb if return_rgb else None,
        'alpha': alpha if return_alpha else None,
        'depth': depth if return_depth else None,
    }

    return ret


def rasterize(
        faces,
        textures,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        background_color=DEFAULT_BACKGROUND_COLOR
):
    """
    Generate RGB images from faces and textures.

    Args:
        faces: see `rasterize_rgbad`.
        textures: see `rasterize_rgbad`.
        image_size: see `rasterize_rgbad`.
        anti_aliasing: see `rasterize_rgbad`.
        near: see `rasterize_rgbad`.
        far: see `rasterize_rgbad`.
        eps: see `rasterize_rgbad`.
        background_color: see `rasterize_rgbad`.

    Returns:
        ~paddle.Tensor: RGB images. The shape is [batch size, 3, image_size, image_size].

    """
    return rasterize_rgbad(
        faces, textures, image_size, anti_aliasing, near, far, eps, background_color, True, False, False)['rgb']


def rasterize_silhouettes(
        faces,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
):
    """
    Generate alpha channels from faces.

    Args:
        faces: see `rasterize_rgbad`.
        image_size: see `rasterize_rgbad`.
        anti_aliasing: see `rasterize_rgbad`.
        near: see `rasterize_rgbad`.
        far: see `rasterize_rgbad`.
        eps: see `rasterize_rgbad`.

    Returns:
        ~paddle.Tensor: Alpha channels. The shape is [batch size, image_size, image_size].

    """
    return rasterize_rgbad(faces, None, image_size, anti_aliasing, near, far, eps, None, False, True, False)['alpha']


def rasterize_depth(
        faces,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
):
    """
    Generate depth images from faces.

    Args:
        faces: see `rasterize_rgbad`.
        image_size: see `rasterize_rgbad`.
        anti_aliasing: see `rasterize_rgbad`.
        near: see `rasterize_rgbad`.
        far: see `rasterize_rgbad`.
        eps: see `rasterize_rgbad`.

    Returns:
        ~paddle.Tensor: Depth images. The shape is [batch size, image_size, image_size].

    """
    return rasterize_rgbad(faces, None, image_size, anti_aliasing, near, far, eps, None, False, False, True)['depth']
