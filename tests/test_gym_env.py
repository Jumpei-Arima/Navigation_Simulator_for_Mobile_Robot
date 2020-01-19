from unittest import TestCase
import os
import shutil
import gym
import nsmr
from nsmr.envs import NsmrGymEnv

print(__file__)

class TestNsmrGymEnv(TestCase):
    def test_main(self):
        env = gym.make("nsmr-v0")
        obs = env.reset()

        # Try stepping a few times
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        env.close()

    def test_randomize(self):
        env = NsmrGymEnv(randomize=True)
        obs = env.reset()

        # Try stepping a few times
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        env.close()

    def test_render(self):
        env = gym.make("nsmr-v0")
        env = NsmrGymEnv()
        obs = env.reset()

        # Try stepping a few times
        for i in range(10):
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        env.close()

    def test_make_collision_map(self):
        shutil.copyfile("./tests/test_map.json", "./nsmr/envs/layouts/test_map.json")
        env = NsmrGymEnv(layout="test_map")
        env.close()
        os.remove("./nsmr/envs/layouts/test_map.json")
        os.remove("./nsmr/envs/layouts/test_map_collision_map.pkl")

if __name__ == '__main__':  # pragma: no cover
    test = TestNsmrGymEnv()
    test.test_main()
    test.test_randomize()
    test.test_render()
    test.test_make_collision_map()
