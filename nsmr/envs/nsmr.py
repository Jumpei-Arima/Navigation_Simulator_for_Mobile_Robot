import os
import json
import hashlib
import pickle

import numpy as np
from tqdm import tqdm

from nsmr.envs.consts import *
from nsmr.envs.obs.raycasting import Raycasting

class NSMR(object):
    def __init__(self, layout=SIMPLE_MAP, randomize=False):
        basedir = os.path.dirname(__file__)
        self.cache_dir_path = os.path.join(basedir, ".nsmr_cache")
        collision_map_file_path = os.path.join(basedir,"layouts", str(layout)+'_collision_map.pkl')
        layout = open(os.path.join(basedir, "layouts", layout + ".json"))
        cfilename = "{}.json".format(hashlib.md5(str(layout).encode()).hexdigest()[:10])
        cached_layout = self.lookup_cache(cfilename)
        cache_found = cached_layout is not False
        if cached_layout:
            self.layout = cached_layout

        self.layout = json.load(layout)

        self.randomize = randomize

        self.dimentions = [self.layout['dimention_x'], self.layout['dimention_y']]

        self.MAP = self.init_map()

        if os.path.exists(collision_map_file_path):
            with open(collision_map_file_path, 'rb') as f:
                self.collision_map = pickle.load(f)
        else:
            self.collision_map = self.make_collision_map()
            with open(collision_map_file_path, 'wb') as f:
                pickle.dump(self.collision_map, f)

        if not cache_found:
            print("Caching layout to: " + cfilename)
            with open(self.get_cache_filename(cfilename), "w") as outfile:
                json.dump(self.layout, outfile, indent=1)
        
        self.pose = self.init_pose()
        self.target = self.init_pose()
        self.lidar = Raycasting(self.MAP)
        self.obs = None

    def init_pose(self):
        return np.zeros(3)
    
    def reset_pose(self):
        while True:
            pose = [np.random.rand()*self.dimentions[0]*RESOLUTION,
                    np.random.rand()*self.dimentions[1]*RESOLUTION,
                    np.random.rand()*2.0*np.pi]
            if not self.is_collision(pose):
                break
        self.pose = pose
        while True:
            pose = [np.random.rand()*self.dimentions[0]*RESOLUTION,
                    np.random.rand()*self.dimentions[1]*RESOLUTION,
                    np.random.rand()*2.0*np.pi]
            dis = np.sqrt((pose[0]-self.pose[0])*(pose[0]-self.pose[0]) + (pose[1]-self.pose[1])*(pose[1]-self.pose[1]))
            if not self.is_collision(pose) or dis < ROBOT_RADIUS:
                break
        self.target = pose
        self.pre_dis = self.get_relative_target_position()[0]

    def update(self, action):
        if self.randomize:
            action[0] += np.random.normal(0, LINEAR_VELOCITY_NOISE)
            action[1] += np.random.normal(0, ANGULAR_VELOCITY_NOISE)
        self.pose[0] += action[0]*np.cos(self.pose[2])*DT
        self.pose[1] += action[0]*np.sin(self.pose[2])*DT
        self.pose[2] += action[1]*DT
        self.pose[2] = self.angle_normalize(self.pose[2])

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

    def is_goal(self):
        return True if self.dis < ROBOT_RADIUS else False
 
    def get_lidar(self):
        obs = np.empty(NUM_LIDAR)
        for i in range(len(obs)):
            angle = i * ANGLE_INCREMENT - MAX_ANGLE
            obs[i] = self.lidar.process(self.pose, angle)
        if self.randomize:
            obs += np.random.normal(0, LIDAR_NOISE)
        self.obs = obs
        return obs

    def get_relative_target_position(self):
        self.dis = np.sqrt((self.target[0]-self.pose[0])*(self.target[0]-self.pose[0]) + (self.target[1]-self.pose[1])*(self.target[1]-self.pose[1]))
        theta = np.arctan2((self.target[1]-self.pose[1]),(self.target[1]-self.pose[1]))
        self.theta = self.angle_diff(theta, self.pose[2])
        return np.array([self.dis, np.sin(self.theta), np.cos(self.theta)])
        
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

    def init_map(self):
        MAP = [[0 for i in range(self.dimentions[1])] for j in range(self.dimentions[0])]
        for i in range(0, self.dimentions[0]-1):
            MAP[i][0] = 1
            MAP[i][self.dimentions[1]-1] = 1
        for j in range(1, self.dimentions[1]-2):
            MAP[0][j] = 1
            MAP[self.dimentions[0]-1][j] = 1
        for obj_info in self.layout['static_objects']:
            typ = obj_info['type']
            if typ == "Block":
                self.make_rect(MAP,obj_info)
        return MAP

    def make_rect(self, MAP, info):
        for i in range(info["l"]-1, info["r"]):
            for j in range(info["t"]-1, info["b"]):
                MAP[i][j] = True
    
    def make_collision_map(self):
        pbar = tqdm(total=100)
        radius = int(ROBOT_RADIUS/RESOLUTION)
        grid = self.MAP
        collision_map = [[False for i in range(self.dimentions[1])] for j in range(self.dimentions[0])]
        for i in range(self.dimentions[0]):
            for j in range(self.dimentions[1]):
                if grid[i][j]:
                    collision_map[i][j] = True
                    for i_ in range(radius*2):
                        for j_ in range(radius*2):
                            x = int(i-radius+i_)
                            y = int(j-radius+j_)
                            dis = np.sqrt((x-i)*(x-i)+(y-j)*(y-j))
                            if 0<x<self.dimentions[0]-1 and 0<y<self.dimentions[1]-1:
                                if dis < radius:
                                    collision_map[x][y] = True
            if i%(self.dimentions[0]/100) == 0:
                pbar.update(1)
        pbar.close()
        return collision_map

    def lookup_cache(self, fname):
        if not os.path.exists(self.cache_dir_path):
            os.mkdir(self.cache_dir_path)
        fname = os.path.join(self.cache_dir_path, fname)
        if not os.path.exists(fname):
            return False
        return open(fname, "r")

    def get_cache_filename(self, fname):
        if not os.path.exists(self.cache_dir_path):
            os.mkdir(self.cache_dir_path)
        fname = os.path.join(self.cache_dir_path, fname)
        return fname