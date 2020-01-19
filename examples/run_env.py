import argparse

import numpy as np
import gym

from nsmr.envs import NsmrGymEnv

if __name__ == "__main__":
    print("Navigation Simulator for Mobile Robot")
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=str, default='simple_map')
    parser.add_argument('--max_episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--randomize', action='store_true')
    args = parser.parse_args()

    env = gym.make("nsmr-v0")

    for i_episode in range(args.max_episodes):
        observation = env.reset()
        done = False
        ep_r = 0
        for t in range(args.max_steps):
            if(not args.no_render):
                env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            # print(observation)
            # print(reward)
            ep_r += reward
            if done:
                print("Episode %d  finished after %d timesteps, reward: %f "% (i_episode+1, t+1,ep_r))
                break
    env.close()
