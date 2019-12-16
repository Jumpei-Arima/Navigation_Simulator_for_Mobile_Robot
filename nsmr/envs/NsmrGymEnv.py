import numpy as np
import gym
from gym import spaces

from nsmr.envs.consts import *
from nsmr.envs.renderer import Renderer
from nsmr.envs.state import State

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
        elif not self.state.is_movable():
            reward = -self.collision_penalty
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
