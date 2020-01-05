from gym.envs.registration import register

from nsmr.envs.NsmrGymEnv import NsmrGymEnv
from nsmr.envs.consts import *

register(
    id='nsmr-v0',
    entry_point='nsmr.envs.NsmrGymEnv:NsmrGymEnv',
    max_episode_steps=1000,
)
