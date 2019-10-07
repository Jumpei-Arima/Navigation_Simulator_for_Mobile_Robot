import os
import json
import hashlib
import pickle

import numpy as np
import gym
from gym import spaces
from tqdm import tqdm

from nsmr.envs.consts import *
from nsmr.envs.obs.raycasting import Raycasting

class NsmrGymEnv(gym.Env):
    def __init__(self,
                 layout=SIMPLE_MAP,
                 render=True,
                 goal_reward=5.0,
                 collision_penalty=5.0,
                 alpha=0.3,
                 beta=0.01,
                 stop_penalty=0.05):
        self.state = State(layout)
        self.observation_space = spaces.Dict(dict(
            lidar=spaces.Box(low=MIN_RANGE, high=MAX_RANGE, shape=(int(NUM_LIDAR/NUM_KERNEL),)),
            target=spaces.Box(np.array([MIN_TARGET_DISTANCE,-1.0,-1.0]), np.array([MAX_TARGET_DISTANCE,1.0,1.0]))
        ))
        self.action_space = spaces.Box(
            np.array([MIN_LINEAR_VELOCITY,-MAX_ANGULAR_VELOCITY]),
            np.array([MAX_LINEAR_VELOCITY,MAX_ANGULAR_VELOCITY]))
        self.renderer = Renderer(self.state)
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.alpha = alpha
        self.beta = beta
        self.stop_penalty = stop_penalty

    def reset(self):
        self.t = 0
        self.state.reset_pose()
        observation = self.get_observation()
        return observation
    
    def step(self, action):
        self.t += 1
        self.state.update(action)
        observation = self.get_observation()
        reward = self.get_reward()
        done = self.is_done()
        info = {}
        return observation, reward, done, info

    def render(self, mode='human'):
        self.renderer.render(self.state, mode)

    def get_observation(self):
        observation = {}
        observation["lidar"] = self.state.get_lidar()
        observation["target"] = self.state.get_relative_target_position()
        return observation

    def get_reward(self):
        if self.state.is_goal():
            reward = self.goal_reward
        elif self.state.is_collision():
            reward = -self.collision_penalty
        else:
            reward = self.alpha*(self.state.pre_dis - self.state.dis)
        if abs(self.state.pre_dis - self.state.dis) < 1e-6:
            reward -= self.stop_penalty
        reward -= self.beta/(2*np.pi)*abs(self.state.theta)
        self.state.pre_dis = self.state.dis
        return reward
    
    def is_done(self):
        done = False
        if self.t >= MAX_STEPS:
            done = True
        if not self.state.is_movable():
            done = True
        if self.state.is_collision():
            done = True
        if self.state.is_goal():
            done = True
        return done

    def close(self):
        self.renderer.close()

class State(object):
    def __init__(self,
                 layout = SIMPLE_MAP):
        basedir = os.path.dirname(__file__)
        self.cache_dir_path = os.path.join(basedir, ".nsmr_cache")
        collision_map_file_path = os.path.join(basedir,"layouts", str(layout)+'_collision_map.pkl')
        layout = open(os.path.join(basedir, "layouts", layout + ".json"))
        cfilename = "{}.json".format(
            hashlib.md5(str(layout).encode()).hexdigest()[:10])
        cached_layout = self.lookup_cache(cfilename)
        cache_found = cached_layout is not False
        if cached_layout:
            print("Cached layout found")
            #self.layout = cached_layout

        self.layout = json.load(layout)

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
        self.pose[0] += action[0]*np.cos(self.pose[2])*DT
        self.pose[1] += action[0]*np.sin(self.pose[2])*DT
        self.pose[2] += action[1]*DT
        self.pose[2] = self.angle_normalize(self.pose[2])

    def is_movable(self, pose=None):
        if pose is None:
            pose = self.pose
        i = int(pose[0]/RESOLUTION)
        j = int(pose[1]/RESOLUTION)
        return (0 <= pose[0] < self.dimentions[0]*RESOLUTION and 0 <= pose[1] < self.dimentions[1]*RESOLUTION and self.MAP[i][j] == 0)

    def is_collision(self, pose=None):
        if pose is None:
            pose = self.pose
        i = int(pose[0]/RESOLUTION)
        j = int(pose[1]/RESOLUTION)
        return self.collision_map[i][j]

    def is_goal(self):
        return True if self.dis < ROBOT_RADIUS else False
 
    def get_lidar(self):
        n = int(NUM_LIDAR/NUM_KERNEL)
        obs = np.empty(n)
        a_n = ANGLE_INCREMENT / (float)(NUM_KERNEL)
        for i in range(n):
            o = []
            _start = i*NUM_KERNEL
            _end = (i+1)*NUM_KERNEL-1
            for j in range(_start, _end):
                angle = j * a_n - MAX_ANGLE
                o.append(self.lidar.process(self.pose,angle))
            obs[i] = np.amin(o)
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

class Renderer(object):
    def __init__(self, state):
        self.viewer = None
        self.margin = 0.2
        screen_size = 600
        world_width_x = state.dimentions[0]*RESOLUTION + self.margin * 2.0
        world_width_y = state.dimentions[1]*RESOLUTION + self.margin * 2.0
        self.scale = screen_size / max(world_width_x, world_width_y)
        self.screen_width = int(world_width_x*self.scale)
        self.screen_height = int(world_width_y*self.scale)

    def render(self, state, mode):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            #wall
            l, r, t, b = self.get_lrtb(0,state.dimentions[0],0,state.dimentions[1])
            wall = rendering.PolyLine([(l,b),(l,t),(r,t),(r,b)],True)
            wall.set_color(0.,0.,0.)
            self.viewer.add_geom(wall)
            #robot
            robot = rendering.make_circle(ROBOT_RADIUS*self.scale)
            self.robot_trans = rendering.Transform()
            robot.add_attr(self.robot_trans)
            robot.set_color(0.0,0.0,1.0)
            self.viewer.add_geom(robot)
            robot_orientation = rendering.make_capsule(ROBOT_RADIUS*self.scale,1.0)
            self.orientation_trans = rendering.Transform()
            robot_orientation.set_color(0.0,1.0,0.0)
            robot_orientation.add_attr(self.orientation_trans)
            self.viewer.add_geom(robot_orientation)
            #target
            target = rendering.make_circle(ROBOT_RADIUS*0.3*self.scale)
            self.target_trans = rendering.Transform()
            target.add_attr(self.target_trans)
            target.set_color(1.0,0.0,0.0)
            self.viewer.add_geom(target)
            #obstract
            for obj in state.layout["static_objects"]:
                l, r, t, b = self.get_lrtb(obj["l"],obj["r"],obj["t"],obj["b"])
                ob = rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
                ob.set_color(0,0,0)
                self.viewer.add_geom(ob)

        robot_x = (self.margin + state.pose[0]) * self.scale
        robot_y = (self.margin + state.pose[1]) * self.scale
        robot_orientation = state.pose[2]
        self.robot_trans.set_translation(robot_x, robot_y)
        self.orientation_trans.set_translation(robot_x,robot_y)
        self.orientation_trans.set_rotation(robot_orientation)
        self.target_trans.set_translation((state.target[0]+self.margin)*self.scale,(state.target[1]+self.margin)*self.scale)
        for i in range(len(state.obs)):
            lidar = rendering.make_capsule(self.scale*state.obs[i],1.0)
            lidar_trans = rendering.Transform()
            lidar_trans.set_translation(robot_x,robot_y)
            lidar_trans.set_rotation(state.pose[2] + i*ANGLE_INCREMENT - MAX_ANGLE)
            lidar.set_color(1.0,0.0,0.0)
            lidar.add_attr(lidar_trans)
            self.viewer.add_onetime(lidar)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def get_lrtb(self, l, r, t, b):
        l = (self.margin+l*RESOLUTION) * self.scale
        r = (self.margin+r*RESOLUTION) * self.scale
        t = (self.margin+t*RESOLUTION) * self.scale
        b = (self.margin+b*RESOLUTION) * self.scale
        return l, r, t, b

    def close(self):
        if self.viewer:
            self.viewer.close()
