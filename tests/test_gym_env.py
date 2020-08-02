from unittest import TestCase
import os
import shutil
import gym
import nsmr
from nsmr.envs import NsmrGymEnv

print(__file__)

class TestNsmrGymEnv(TestCase):
    def test_main(self, env_name):
        env = gym.make(env_name)
        obs = env.reset()
        # Try stepping a few times
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        env.close()

    def test_render(self, env_name):
        env = gym.make(env_name)
        obs = env.reset()
        # Try stepping a few times
        for i in range(10):
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        env.close()

    def test_reward_params(self, env_name):
        env = gym.make(env_name)
        params={"goal_reward": 5.0,
                "collision_penalty": 5.0,
                "alpha": 0.3,
                "beta": 0.01,
                "stop_penalty": 0.05}
        env.set_reward_params(params)
        obs = env.reset()
        # Try stepping a few times
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        env.close()

    def test_make_collision_map(self, env_name):
        shutil.copyfile("./tests/test_map.json", "./nsmr/envs/layouts/test_map.json")
        env = gym.make(env_name)
        env.set_layout("test_map")
        env.close()
        os.remove("./nsmr/envs/layouts/test_map.json")
        os.remove("./nsmr/envs/layouts/test_map_collision_map.pkl")

if __name__ == '__main__':  # pragma: no cover
    test = TestNsmrGymEnv()
    env_names = ["nsmr-v0", "nsmr-v1", "NsmrSimple-v0"]
    for env_name in env_names:
        print(env_name)
        test.test_main(env_name)
        test.test_render(env_name)
        test.test_reward_params(env_name)
        test.test_make_collision_map(env_name)
