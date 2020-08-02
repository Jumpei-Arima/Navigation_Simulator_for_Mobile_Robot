from unittest import TestCase
import os
import shutil
import gym
import nsmr

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

    def test_render(self):
        env = gym.make("nsmr-v0")
        obs = env.reset()
        # Try stepping a few times
        for i in range(10):
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        env.close()

    def test_reward_params(self):
        env = gym.make("nsmr-v0")
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

    def test_make_collision_map(self):
        shutil.copyfile("./tests/test_map.json", "./nsmr/envs/layouts/test_map.json")
        env = gym.make("nsmr-v0")
        env.set_layout("test_map")
        env.close()
        os.remove("./nsmr/envs/layouts/test_map.json")
        os.remove("./nsmr/envs/layouts/test_map_collision_map.pkl")

if __name__ == '__main__':  # pragma: no cover
    test = TestNsmrGymEnv()
    test.test_main()
    test.test_render()
    test.test_reward_params()
    test.test_make_collision_map()
