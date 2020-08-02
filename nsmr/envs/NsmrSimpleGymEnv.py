import numpy as np
import gym
from gym import spaces

from nsmr.envs.consts import *
from nsmr.envs.renderer import Renderer
from nsmr.envs.nsmr import NSMR

class NsmrSimpleGymEnv(gym.Env):
    def __init__(self,
                 layout=NO_OBJECT,
                 reward_params={"goal_reward": 5.0,
                                "alpha": 0.3,
                                "beta": 0.01,
                                "stop_penalty": 0.05},
                 ):
        # simulator
        self.nsmr = NSMR(layout)

        # gym space
        self.observation_space = spaces.Dict(dict(
            pose=spaces.Box(np.array([-10,-10,-3.141592]), np.array([10,10,3.141592])),
            target=spaces.Box(np.array([-10,-10,-3.141592]), np.array([10,10,3.141592]))
        ))
        self.action_space = spaces.Box(
            np.array([MIN_LINEAR_VELOCITY,MIN_ANGULAR_VELOCITY]),
            np.array([MAX_LINEAR_VELOCITY,MAX_ANGULAR_VELOCITY]))

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
        info = {}

        return observation, reward, done, info

    def render(self, mode='human'):
        self.renderer.render(self.nsmr, mode)

    def get_observation(self):
        observation = {}
        observation["pose"] = self.nsmr.pose
        observation["target"] = self.nsmr.target
        return observation

    def get_reward(self, observation):
        target_info = self.nsmr.get_relative_target_position()
        dis = target_info[0]
        ddis = self.pre_dis - dis
        theta = np.arccos(target_info[2])
        if dis < ROBOT_RADIUS:
            reward = self.reward_params["goal_reward"]
            self.goal = True
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
        if self.goal:
            done = True
        return done

    def close(self):
        self.renderer.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
