import os
import json
import hashlib
import pickle

import numpy as np

from nsmr.obs.raycasting import Raycasting
from nsmr.utils.utils import *

class NSMR(object):
    def __init__(self,
                 robot="robot",
                 layout="simple_map",
                 dt=0.1):
        self.set_config(robot, layout)
        
        self.DT = dt

    def set_config(self, robot, layout):
        if os.path.exists(robot):
            with open(robot) as f:
                self.robot = json.load(f)
        else:
            file_path = os.path.join(
                os.path.dirname(__file__), "../robots", str(robot) + ".json")
            if os.path.exists(file_path):
                with open(file_path) as f:
                    self.robot = json.load(f)
            else:
                self.robot = None
        self.set_layout(layout)

        self.obs = None
        self.pre_action = np.zeros(2)
        self.reset_noise_param()

    def set_layout(self, layout):
        self.layout = self.get_layout(layout)
        self.dimentions = [self.layout['dimention_x'], self.layout['dimention_y']]
        self.MAP = self.init_map()
        self.collision_map = self.get_collision_map(layout)
        self.lidar = Raycasting(self.MAP,
                                self.layout['resolution'],
                                self.robot["lidar"]["max_range"],
                                self.robot["lidar"]["min_range"])

    def reset_pose(self):
        while True:
            pose = self.get_random_pose()
            if not self.is_collision(pose):
                break
        self.pose = pose
        while True:
            target = self.get_random_pose()
            dis = get_dis(self.pose, target)
            if not self.is_collision(target) or dis < self.robot["radius"]:
                break
        self.target = target

    def reset_noise_param(self):
        self.distance_until_noise = np.random.exponential(1.0/(1e-100 + self.robot["noise"]["noise_per_meter"]))
        self.time_until_stuck = np.random.exponential(self.robot["noise"]["expected_stuck_time"])
        self.time_until_escape = np.random.exponential(self.robot["noise"]["expected_escape_time"])
        self.is_stuck = False
        self.lidar_pos_noise = np.array([np.random.normal(0.0, self.robot["lidar"]["noise"]["position"]),
                                         np.random.normal(0.0, self.robot["lidar"]["noise"]["position"]),
                                         np.random.normal(0.0, self.robot["lidar"]["noise"]["angle"])
                                        ])

    def update(self, action):
        action = self.add_bias(action)
        action = self.stuck(action)
        action = self.check_action(action)
        self.state_transition(action)
        self.add_noise(action)
        self.pre_action = action
    
    def check_action(self, action):
        action[0] = max(self.pre_action[0]-self.robot["max_linear_acceleration"]*self.DT,
                        min(self.pre_action[0]+self.robot["max_linear_acceleration"]*self.DT, action[0]))
        action[1] = max(self.pre_action[1]-self.robot["max_angular_acceleration"]*self.DT,
                        min(self.pre_action[1]+self.robot["max_angular_acceleration"]*self.DT, action[1]))
        action[0] = max(self.robot["min_linear_velocity"], min(self.robot["max_linear_velocity"], action[0]))
        action[1] = max(self.robot["min_angular_velocity"], min(self.robot["max_angular_velocity"], action[1]))
        return action

    def state_transition(self, action):
        pre_theta = self.pose[2]
        if abs(action[1])<1e-10:
            self.pose[0] += action[0]*np.cos(pre_theta)*self.DT
            self.pose[1] += action[0]*np.sin(pre_theta)*self.DT
            self.pose[2] += action[1]*self.DT
        else:
            self.pose[0] += action[0]/action[1]*(np.sin(pre_theta+action[1]*self.DT)-np.sin(pre_theta))
            self.pose[1] += action[0]/action[1]*(-np.cos(pre_theta+action[1]*self.DT)+np.cos(pre_theta))
            self.pose[2] += action[1]*self.DT
        self.pose[2] = angle_normalize(self.pose[2])

    def add_noise(self, action):
        self.distance_until_noise -= abs(action[0])*self.DT + self.robot["radius"]*abs(action[1])*self.DT
        if self.distance_until_noise <= 0:
            self.distance_until_noise += np.random.exponential(1.0/(1e-100 + self.robot["noise"]["noise_per_meter"]))
            self.pose[2] += np.random.normal(0, self.robot["noise"]["angle"])

    def add_bias(self, action):
        action[0] *= np.random.normal(1.0, self.robot["noise"]["linear_velocity"])
        action[1] *= np.random.normal(1.0, self.robot["noise"]["angular_velocity"])
        return action

    def stuck(self, action):
        if self.is_stuck:
            self.time_until_escape -= self.DT
            if self.time_until_escape <= 0.0:
                self.time_until_escape += np.random.exponential(self.robot["noise"]["expected_escape_time"])
                self.is_stuck = False
        else:
            self.time_until_stuck -= self.DT
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += np.random.exponential(self.robot["noise"]["expected_stuck_time"])
                self.is_stuck = True
        action[0] *= (not self.is_stuck)
        action[1] *= (not self.is_stuck)
        return action

    def available_index(self, i, j):
        return (0<=i<self.dimentions[0] and 0<=j<self.dimentions[1])
    
    def is_collision(self, pose=None):
        if pose is None:
            pose = self.pose
        i = int(pose[0]/self.layout['resolution'])
        j = int(pose[1]/self.layout['resolution'])
        if self.available_index(i, j):
            return self.collision_map[i][j]
        else:
            return False

    def get_lidar(self, pose=None):
        if pose is None:
            pose = self.pose.copy()
        pose += self.lidar_pos_noise
        pose[2] = angle_normalize(pose[2])
        obs = np.empty(self.robot["lidar"]["num_range"])
        for i in range(len(obs)):
            angle = i * self.robot["lidar"]["angle_increment"] - self.robot["lidar"]["max_angle"]
            obs[i] = self.lidar.process(pose.tolist(), angle)
            if np.random.rand() < self.robot["lidar"]["noise"]["oversight_prob"]:
                obs[i] = np.random.rand()*(self.robot["lidar"]["max_range"]-self.robot["lidar"]["min_range"]) + self.robot["lidar"]["min_range"]
        obs += np.random.normal(0, self.robot["lidar"]["noise"]["range"])
        self.obs = obs
        return obs

    def get_relative_target_position(self, pose=None, target=None):
        if pose is None:
            pose = self.pose
        if target is None:
            target = self.target
        dis = get_dis(target, pose)
        theta = np.arctan2((target[1]-pose[1]),(target[0]-pose[0]))
        theta = angle_diff(theta, pose[2])
        return np.array([dis, np.sin(theta), np.cos(theta)])

    def get_random_pose(self):
        scale = np.array([self.dimentions[0]*self.layout['resolution'],
                          self.dimentions[1]*self.layout['resolution'],
                          2.0*np.pi])
        pose = np.random.rand(3)*scale
        return pose

    def get_layout(self, layout):
        if os.path.exists(layout):
            with open(layout) as f:
                layout = json.load(f)
        else:
            file_path = os.path.join(
                os.path.dirname(__file__), "../layouts", str(layout) + ".json")
            if os.path.exists(file_path):
                with open(file_path) as f:
                    layout = json.load(f)
            else:
                layout = None
        return layout
    
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

    def get_collision_map(self, layout):
        if os.path.exists(layout):
            file_path = os.path.join(
                os.path.split(layout)[0],
                os.path.splitext(os.path.basename(layout))[0] + "_collision_map.pkl")
        else:
            file_path = os.path.join(
                os.path.dirname(__file__), "../layouts",
                str(layout) + "_collision_map.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                collision_map = pickle.load(f)
        else:
            collision_map = self.make_collision_map()
            with open(file_path, 'wb') as f:
                pickle.dump(collision_map, f)
        return collision_map

    def make_collision_map(self):
        radius = int(self.robot["radius"]/self.layout['resolution'])
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