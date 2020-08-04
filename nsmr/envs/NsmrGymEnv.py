import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from nsmr.envs.consts import *
from nsmr.envs.renderer import Renderer
from nsmr.envs.nsmr import NSMR

class NsmrGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self,
                 layout=SIMPLE_MAP,
                 reward_params={"goal_reward": 5.0,
                                "collision_penalty": 5.0,
                                "alpha": 0.3,
                                "beta": 0.01,
                                "stop_penalty": 0.05},
                 ):
        # simulator
        self.nsmr = NSMR(layout)

        # gym space
        self.observation_space = spaces.Dict(dict(
            lidar=spaces.Box(low=MIN_RANGE, high=MAX_RANGE, shape=(NUM_LIDAR,), dtype=np.float32),
            target=spaces.Box(np.array([MIN_TARGET_DISTANCE,-1.0,-1.0]), np.array([MAX_TARGET_DISTANCE,1.0,1.0]), dtype=np.float32)
        ))
        self.action_space = spaces.Box(
            low = np.array([MIN_LINEAR_VELOCITY,MIN_ANGULAR_VELOCITY]),
            high = np.array([MAX_LINEAR_VELOCITY,MAX_ANGULAR_VELOCITY]),
            dtype = np.float32
            )

        # renderer
        self.renderer = Renderer(self.nsmr.dimentions)

        # reward params
        self.reward_params = reward_params

        self.reset()

    def set_reward_params(self, reward_params):
        self.reward_params = reward_params
        self.reset()

    def set_layout(self, layout):
        self.nsmr.set_layout(layout)
        self.renderer = Renderer(self.nsmr.dimentions)
        self.reset()

    def reset(self):
        self.t = 0
        self.nsmr.reset_pose()
        self.nsmr.reset_noise_param()
        observation = self.get_observation()
        self.pre_dis = observation["target"][0]
        self.goal = False
        return observation
    
    def step(self, action):
        self.t += 1
        self.nsmr.update(action)
        observation = self.get_observation()
        reward = self.get_reward(observation)
        done = self.is_done()
        info = {"pose": self.nsmr.pose, "target": self.nsmr.target}

        return observation, reward, done, info

    def render(self, mode='human'):
        return self.renderer.render(self.nsmr, mode)

    def get_observation(self):
        observation = {}
        observation["lidar"] = self.nsmr.get_lidar()
        observation["target"] = self.nsmr.get_relative_target_position()
        return observation

    def get_reward(self, observation):
        dis = observation["target"][0]
        ddis = self.pre_dis - dis
        theta = np.arccos(observation["target"][2])
        if dis < ROBOT_RADIUS:
            reward = self.reward_params["goal_reward"]
            self.goal = True
        elif self.nsmr.is_collision():
            reward = -self.reward_params["collision_penalty"]
        else:
            reward = self.reward_params["alpha"] * ddis
        if abs(ddis) < 1e-6:
            reward -= self.reward_params["stop_penalty"]
        reward -= self.reward_params["beta"]/(2*np.pi)*abs(theta)
        self.pre_dis = dis
        return reward
    
    def is_done(self):
        done = False
        if self.t >= MAX_STEPS:
            done = True
        if self.nsmr.is_collision():
            done = True
        if self.goal:
            done = True
        return done

    def close(self):
        self.renderer.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
