#!/usr/bin/env python3

import os
import numpy as np
import gym
import nsmr

env = gym.make("nsmr-v0")
obs = env.reset()

# Try stepping a few times
for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
