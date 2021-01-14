from gym.envs.registration import register

from nsmr.envs.NsmrGymEnv import NsmrGymEnv
from nsmr.envs.NsmrSimpleGymEnv import NsmrSimpleGymEnv
from nsmr.envs.NsmrMinPoolingGymEnv import NsmrMinPoolingGymEnv
from nsmr.envs.consts import *

register(
    id='Nsmr-v1',
    entry_point='nsmr.envs.NsmrGymEnv:NsmrGymEnv',
    max_episode_steps=500,
)

register(
    id='NsmrMinPooling-v1',
    entry_point='nsmr.envs.NsmrMinPoolingGymEnv:NsmrMinPoolingGymEnv',
    max_episode_steps=500,
)

register(
    id='NsmrSimple-v1',
    entry_point='nsmr.envs.NsmrSimpleGymEnv:NsmrSimpleGymEnv',
    max_episode_steps=500,
)