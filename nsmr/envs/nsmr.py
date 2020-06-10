import os
import json
import hashlib
import pickle

import numpy as np

from nsmr.envs.consts import *
from nsmr.envs.obs.raycasting import Raycasting

class NSMR(object):
    def __init__(self, layout=SIMPLE_MAP):
        self.set_layout(layout)
        
        self.lidar = Raycasting(self.MAP)
        self.obs = None

        self.reset_noise_param()

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

    def reset_noise_param(self):
        self.distance_until_noise = np.random.exponential(1.0/(1e-100 + NOISE_PER_METER))
        self.time_until_stuck = np.random.exponential(EXPECTED_STUCK_TIME)
        self.time_until_escape = np.random.exponential(EXPECTED_ESCAPE_TIME)
        self.is_stuck = False
        self.lidar_pos_noise = [np.random.normal(0.0, LIDAR_POSE_NOISE),
                                np.random.normal(0.0, LIDAR_POSE_NOISE),
                                np.random.normal(0.0, LIDAR_ORIENTATION_NOISE)]

    def update(self, action):
        action = self.add_bias(action)
        action = self.stuck(action)
        self.state_transition(action)
        self.add_noise(action)

    def state_transition(self, action):
        pre_theta = self.pose[2]
        if abs(action[1])<1e-10:
            self.pose[0] += action[0]*np.cos(pre_theta)*DT
            self.pose[1] += action[0]*np.sin(pre_theta)*DT
            self.pose[2] += action[1]*DT
        else:
            self.pose[0] += action[0]/action[1]*(np.sin(pre_theta+action[1]*DT)-np.sin(pre_theta))
            self.pose[1] += action[0]/action[1]*(-np.cos(pre_theta+action[1]*DT)+np.cos(pre_theta))
            self.pose[2] += action[1]*DT
        self.pose[2] = self.angle_normalize(self.pose[2])

    def add_noise(self, action):
        self.distance_until_noise -= abs(action[0])*DT + ROBOT_RADIUS*abs(action[1])*DT
        if self.distance_until_noise <= 0:
            self.distance_until_noise += np.random.exponential(1.0/(1e-100 + NOISE_PER_METER))
            self.pose[2] += np.random.normal(0, ANGLE_NOISE)

    def add_bias(self, action):
        action[0] *= np.random.normal(1.0, LINEAR_VELOCITY_NOISE)
        action[1] *= np.random.normal(1.0, ANGULAR_VELOCITY_NOISE)
        return action

    def stuck(self, action):
        if self.is_stuck:
            self.time_until_escape -= DT
            if self.time_until_escape <= 0.0:
                self.time_until_escape += np.random.exponential(EXPECTED_ESCAPE_TIME)
                self.is_stuck = False
        else:
            self.time_until_stuck -= DT
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += np.random.exponential(EXPECTED_STUCK_TIME)
                self.is_stuck = True
        action[0] *= (not self.is_stuck)
        action[1] *= (not self.is_stuck)
        return action

    def set_layout(self, layout):
        self.layout = self.get_layout(layout)
        self.dimentions = [self.layout['dimention_x'], self.layout['dimention_y']]
        self.MAP = self.init_map()
        self.collision_map = self.get_collision_map(layout)
        self.lidar = Raycasting(self.MAP)

    def available_index(self, i, j):
        return (0<=i<self.dimentions[0] and 0<=j<self.dimentions[1])
    
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
            pose = self.pose.copy()
        pose[0] += self.lidar_pos_noise[0]
        pose[1] += self.lidar_pos_noise[1]
        pose[2] += self.lidar_pos_noise[2]
        obs = np.empty(NUM_LIDAR)
        for i in range(len(obs)):
            angle = i * ANGLE_INCREMENT - MAX_ANGLE
            obs[i] = self.lidar.process(pose, angle)
            if np.random.rand() < OVERSIGHT_PROB:
                obs[i] = np.random.rand()*(MAX_RANGE-MIN_RANGE) + MIN_RANGE
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
        theta = self.angle_diff(theta, pose[2])
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

