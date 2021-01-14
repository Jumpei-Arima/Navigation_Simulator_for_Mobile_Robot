import math

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from nsmr.envs.renderer import Renderer
from nsmr.envs.NsmrGymEnv import NsmrGymEnv

class NsmrMinPoolingGymEnv(NsmrGymEnv):
    def __init__(self,
                 robot="robot",
                 layout="simple_map",
                 reward_params={"goal_reward": 5.0,
                                "collision_penalty": 5.0,
                                "alpha": 0.3,
                                "beta": 0.01,
                                "stop_penalty": 0.05},
                 max_steps=500,
                 kernel_size=20
                 ):
        self.kernel_size = kernel_size
        super(NsmrMinPoolingGymEnv, self).__init__(robot=robot,
                                                   layout=layout,
                                                   reward_params=reward_params,
                                                   max_steps=max_steps)
        self.lidar_size = int(self.nsmr.robot["lidar"]["num_range"] / kernel_size)

        # gym space
        self.observation_space = spaces.Dict(dict(
            lidar=spaces.Box(low=self.nsmr.robot["lidar"]["min_range"],
                             high=self.nsmr.robot["lidar"]["max_range"],
                             shape=(self.lidar_size,),
                             dtype=np.float32),
            target=spaces.Box(np.array([0.0,-1.0,-1.0]),
                              np.array([100.0,1.0,1.0]),
                              dtype=np.float32)
        ))

        self.reset()

    def get_observation(self):
        observation = {}
        observation["lidar"] = self.minpooling(self.nsmr.get_lidar())
        observation["target"] = self.nsmr.get_relative_target_position()
        return observation

    def minpooling(self, x):
        x = x.reshape([-1, self.kernel_size])
        x = np.amin(x, axis=1)
        return x