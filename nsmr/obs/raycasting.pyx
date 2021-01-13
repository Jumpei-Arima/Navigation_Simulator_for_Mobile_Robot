import math

cimport cython
from libc.math cimport sqrt, sin, cos
from libc.math cimport abs as cabs

from nsmr.envs.consts import *

@cython.boundscheck(False)
@cython.wraparound(False)

class Raycasting():
    def __init__(self, list MAP):
        self.MAP_RESO = 1.0 / RESOLUTION
        self.MAP_RESOLUTION = RESOLUTION
        self.MAP = MAP
        self.MAP_SIZE = [len(self.MAP), len(self.MAP[0])]

    def process(self, list pose, double angle, list MAP=None):
        if MAP is not None:
            self.MAP= MAP
        cdef int x0, y0, x1, y1, dx, dy, error, derror, x_step, y_step, x, y, x_limit, _x, _y
        cdef list pose_ = [0 for _ in range(3)]
        cdef steep
        x0 = int(pose[0]*self.MAP_RESO)
        y0 = int(pose[1]*self.MAP_RESO)
        x1 = int((pose[0]+MAX_RANGE * cos(pose[2]+angle))*self.MAP_RESO)
        y1 = int((pose[1]+MAX_RANGE * sin(pose[2]+angle))*self.MAP_RESO)
        steep = False
        if cabs(y1-y0) > cabs(x1-x0):
            steep = True
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        dx, dy = cabs(x1-x0), cabs(y1-y0)
        error, derror = 0, dy
        x, y = x0, y0
        x_step, y_step = -1, -1
        if x0<x1:
            x_step = 1
        if y0<y1:
            y_step = 1
        if steep:
            pose_[0] = y
            pose_[1] = x
            if not self.is_movable_grid(pose_):
                _x = (x-x0)*(x-x0)
                _y = (y-y0)*(y-y0)
                return sqrt(_x + _y) * self.MAP_RESOLUTION
        else:
            pose_[0] = x
            pose_[1] = y
            if not self.is_movable_grid(pose_):
                _x = (x-x0)*(x-x0)
                _y = (y-y0)*(y-y0)
                return sqrt(_x + _y) * self.MAP_RESOLUTION
        x_limit = x1 + x_step
        while x != x_limit:
            x = x + x_step
            error = error + derror
            if 2.0*error >= dx:
                y = y + y_step
                error = error - dx
            if steep:
                pose_[0] = y
                pose_[1] = x
                if not self.is_movable_grid(pose_):
                    _x = (x-x0)*(x-x0)
                    _y = (y-y0)*(y-y0)
                    return sqrt(_x + _y) * self.MAP_RESOLUTION
            else:
                pose_[0] = x
                pose_[1] = y
                if not self.is_movable_grid(pose_):
                    _x = (x-x0)*(x-x0)
                    _y = (y-y0)*(y-y0)
                    return sqrt(_x + _y) * self.MAP_RESOLUTION
        return MAX_RANGE

    def is_movable_grid(self, list pose):
        i = int(pose[0])
        j = int(pose[1])
        return (0 <= pose[0] < self.MAP_SIZE[0] and 0 <= pose[1] < self.MAP_SIZE[1] and self.MAP[i][j] == 0)