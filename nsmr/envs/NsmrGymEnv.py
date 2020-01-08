import numpy as np
import gym
from gym import spaces

from nsmr.envs.consts import *
from nsmr.envs.renderer import Renderer
from nsmr.envs.nsmr import NSMR

class NsmrGymEnv(gym.Env):
    def __init__(self,
                 layout=SIMPLE_MAP,
                 render=True,
                 reward_params={"goal_reward": 5.0,
                                "collision_penalty": 5.0,
                                "alpha": 0.3,
                                "beta": 0.01,
                                "stop_penalty": 0.05},
                 randomize=False
                 ):
        # simulator
        self.nsmr = NSMR(layout, randomize)

        # gym space
        self.observation_space = spaces.Dict(dict(
            lidar=spaces.Box(low=MIN_RANGE, high=MAX_RANGE, shape=(NUM_LIDAR,)),
            target=spaces.Box(np.array([MIN_TARGET_DISTANCE,-1.0,-1.0]), np.array([MAX_TARGET_DISTANCE,1.0,1.0]))
        ))
        self.action_space = spaces.Box(
            np.array([MIN_LINEAR_VELOCITY,MIN_ANGULAR_VELOCITY]),
            np.array([MAX_LINEAR_VELOCITY,MAX_ANGULAR_VELOCITY]))

        # renderer
        self.renderer = Renderer(self.nsmr)

        # reward params
        self.goal_reward = reward_params["goal_reward"]
        self.collision_penalty = reward_params["collision_penalty"]
        self.alpha = reward_params["alpha"]
        self.beta = reward_params["beta"]
        self.stop_penalty = reward_params["stop_penalty"]

    def reset(self):
        self.t = 0
        self.nsmr.reset_pose()
        observation = self.get_observation()
        return observation
    
    def step(self, action):
        self.t += 1
        self.nsmr.update(action)
        observation = self.get_observation()
        reward = self.get_reward()
        done = self.is_done()
        info = {}
        return observation, reward, done, info

    def render(self, mode='human'):
        self.renderer.render(self.nsmr, mode)

    def get_observation(self):
        observation = {}
        observation["lidar"] = self.nsmr.get_lidar()
        observation["target"] = self.nsmr.get_relative_target_position()
        return observation

    def get_reward(self):
        if self.nsmr.is_goal():
            reward = self.goal_reward
        elif not self.nsmr.is_movable():
            reward = -self.collision_penalty
        elif self.nsmr.is_collision():
            reward = -self.collision_penalty
        else:
            reward = self.alpha*(self.nsmr.pre_dis - self.nsmr.dis)
        if abs(self.nsmr.pre_dis - self.nsmr.dis) < 1e-6:
            reward -= self.stop_penalty
        reward -= self.beta/(2*np.pi)*abs(self.nsmr.theta)
        self.nsmr.pre_dis = self.nsmr.dis
        return reward
    
    def is_done(self):
        done = False
        if self.t >= MAX_STEPS:
            done = True
        if not self.nsmr.is_movable():
            done = True
        if self.nsmr.is_collision():
            done = True
        if self.nsmr.is_goal():
            done = True
        return done

    def close(self):
        self.renderer.close()
