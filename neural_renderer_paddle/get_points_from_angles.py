from __future__ import division
import math

import paddle

def get_points_from_angles(distance, elevation, azimuth, degrees=True):
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = math.radians(elevation)
            azimuth = math.radians(azimuth)
        return (
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
            -distance * math.cos(elevation) * math.cos(azimuth))
    else:
        if degrees:
            elevation = math.pi/180. * elevation
            azimuth = math.pi/180. * azimuth
    #
        return paddle.stack([
            distance * paddle.cos(elevation) * paddle.sin(azimuth),
            distance * paddle.sin(elevation),
            -distance * paddle.cos(elevation) * paddle.cos(azimuth)
            ]).swapaxes(1,0)
