import os
import json
import hashlib
import pickle

import numpy as np

from nsmr.envs.consts import *
from nsmr.envs.obs.raycasting import Raycasting

class NSMR(object):
    def __init__(self, layout=SIMPLE_MAP, randomize=False):
        self.randomize = randomize
        self.set_layout(layout)
        
        self.lidar = Raycasting(self.MAP)
        self.obs = None

    def reset_pose(self):
        while True:
            pose = self.get_random_pose()
            if not self.is_collision(pose):
                break
        self.pose = pose
        while True:
            target = self.get_random_pose()
            dis = self.get_dis(self.pose, target)
            if not self.is_collision(target) or dis < ROBOT_RADIUS:
                break
        self.target = target

    def update(self, action):
        if self.randomize:
            action[0] += np.random.normal(0, LINEAR_VELOCITY_NOISE)
            action[1] += np.random.normal(0, ANGULAR_VELOCITY_NOISE)
        self.pose[0] += action[0]*np.cos(self.pose[2])*DT
        self.pose[1] += action[0]*np.sin(self.pose[2])*DT
        self.pose[2] += action[1]*DT
        self.pose[2] = self.angle_normalize(self.pose[2])

    def set_layout(self, layout):
        self.layout = self.get_layout(layout)
        self.dimentions = [self.layout['dimention_x'], self.layout['dimention_y']]
        self.MAP = self.init_map()
        self.collision_map = self.get_collision_map(layout)
        self.lidar = Raycasting(self.MAP)

    def available_index(self, i, j):
        return (0<=i<self.dimentions[0] and 0<=j<self.dimentions[1])
    
    def is_movable(self, pose=None):
        if pose is None:
            pose = self.pose
        i = int(pose[0]/RESOLUTION)
        j = int(pose[1]/RESOLUTION)
        if self.available_index(i, j):
            return self.MAP[i][j] == 0
        else:
            return False

    def is_collision(self, pose=None):
        if pose is None:
            pose = self.pose
        i = int(pose[0]/RESOLUTION)
        j = int(pose[1]/RESOLUTION)
        if self.available_index(i, j):
            return self.collision_map[i][j]
        else:
            return False

    def get_lidar(self, pose=None):
        if pose is None:
            pose = self.pose
        obs = np.empty(NUM_LIDAR)
        for i in range(len(obs)):
            angle = i * ANGLE_INCREMENT - MAX_ANGLE
            obs[i] = self.lidar.process(pose, angle)
        if self.randomize:
            obs += np.random.normal(0, LIDAR_NOISE)
        self.obs = obs
        return obs

    def get_relative_target_position(self, pose=None, target=None):
        if pose is None:
            pose = self.pose
        if target is None:
            target = self.target
        dis = self.get_dis(target, pose)
        theta = np.arctan2((target[1]-pose[1]),(target[0]-pose[0]))
        theta = self.angle_diff(theta, self.pose[2])
        return np.array([dis, np.sin(theta), np.cos(theta)])

    def angle_normalize(self, z):
        return np.arctan2(np.sin(z), np.cos(z))

    def angle_diff(self, a, b):
        a = self.angle_normalize(a)
        b = self.angle_normalize(b)
        d1 = a-b
        d2 = 2.0 * np.pi - abs(d1)
        if d1 > 0.0:
            d2 *= -1.0
        if abs(d1) < abs(d2):
            return d1
        else:
            return d2

    def get_random_pose(self):
        pose = [np.random.rand()*self.dimentions[0]*RESOLUTION,
                np.random.rand()*self.dimentions[1]*RESOLUTION,
                np.random.rand()*2.0*np.pi]
        return pose

    def get_dis(self, pa, pb):
        dis = np.sqrt((pa[0]-pb[0])*(pa[0]-pb[0]) + (pa[1]-pb[1])*(pa[1]-pb[1]))
        return dis
    
    def init_map(self):
        MAP = [[False for i in range(self.dimentions[1])] for j in range(self.dimentions[0])]
        # make wall
        for i in range(0, self.dimentions[0]-1):
            MAP[i][0] = True
            MAP[i][self.dimentions[1]-1] = True
        for j in range(1, self.dimentions[1]-2):
            MAP[0][j] = True
            MAP[self.dimentions[0]-1][j] = True
        # make obj
        for obj_info in self.layout['static_objects']:
            typ = obj_info['type']
            if typ == "Block":
                self.make_rect(MAP,obj_info)
        return MAP

    def make_rect(self, MAP, info):
        for i in range(info["l"]-1, info["r"]):
            for j in range(info["t"]-1, info["b"]):
                if self.available_index(i,j):
                    MAP[i][j] = True
    
    def make_rect_collision_map(self, MAP, info, radius):
        for i in range(info["l"]-1-radius, info["r"]+radius):
            for j in range(info["t"]-1-radius, info["b"]+radius):
                if self.available_index(i,j):
                    MAP[i][j] = True
        for i in range(info["l"]-1-radius, info["l"]):
            for j in range(info["t"]-1-radius, info["t"]):
                dis = np.sqrt((info["l"]-i)*(info["l"]-i)+(info["t"]-j)*(info["t"]-j))
                if dis > radius:
                    MAP[i][j] = False
            for j in range(info["b"], info["b"]+radius):
                dis = np.sqrt((info["l"]-i)*(info["l"]-i)+(info["b"]-j)*(info["b"]-j))
                if dis > radius:
                    MAP[i][j] = False
        for i in range(info["r"], info["r"]+radius):
            for j in range(info["t"]-1-radius, info["t"]):
                dis = np.sqrt((info["r"]-i)*(info["r"]-i)+(info["t"]-j)*(info["t"]-j))
                if dis > radius:
                    MAP[i][j] = False
            for j in range(info["b"], info["b"]+radius):
                dis = np.sqrt((info["r"]-i)*(info["r"]-i)+(info["b"]-j)*(info["b"]-j))
                if dis > radius:
                    MAP[i][j] = False

    def make_collision_map(self):
        radius = int(ROBOT_RADIUS/RESOLUTION)
        collision_map = [[False for i in range(self.dimentions[1])] for j in range(self.dimentions[0])]
        # make wall
        for i in range(0, self.dimentions[0]-1):
            for j in range(0, radius-1):
                collision_map[i][j] = True
                collision_map[i][self.dimentions[1]-1-j] = True
        for j in range(1, self.dimentions[1]-2):
            for i in range(0, radius-1):
                collision_map[i][j] = True
                collision_map[self.dimentions[0]-1-i][j] = True
        # make obj
        for obj_info in self.layout['static_objects']:
            typ = obj_info['type']
            if typ == "Block":
                self.make_rect_collision_map(collision_map, obj_info, radius)
        return collision_map

    def get_collision_map(self, layout):
        file_path = os.path.join(os.path.dirname(__file__), "layouts", str(layout) + "_collision_map.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                collision_map = pickle.load(f)
        else:
            collision_map = self.make_collision_map()
            with open(file_path, 'wb') as f:
                pickle.dump(collision_map, f)
        return collision_map

    def get_layout(self, layout):
        file_path = os.path.join(os.path.dirname(__file__), "layouts", str(layout) + ".json")
        with open(file_path) as f:
            layout = json.load(f)
        return layout

