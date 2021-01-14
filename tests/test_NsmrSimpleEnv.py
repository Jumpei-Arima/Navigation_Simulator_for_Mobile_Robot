from unittest import TestCase
import os
import shutil
import gym
import nsmr

print(__file__)

class TestNsmrSimpleGymEnv(TestCase):
    def test_main(self):
        env = gym.make("NsmrSimple-v1")
        obs = env.reset()
        # Try stepping a few times
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        env.close()

    def test_render(self):
        env = gym.make("NsmrSimple-v1")
        obs = env.reset()
        # Try stepping a few times
        for i in range(10):
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        env.close()

    def test_reward_params(self):
        env = gym.make("NsmrSimple-v1")
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

    def test_set_config(self):
        base_path = os.path.dirname(__file__)
        env = gym.make("NsmrSimple-v1")
        env.set_env_config(robot=os.path.join(base_path, "test_robot.json"),
                           layout=os.path.join(base_path, "test_map.json"))
        obs = env.reset()
        # Try stepping a few times
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
        env.close()
        os.remove(os.path.join(base_path, "test_map_collision_map.pkl"))

if __name__ == '__main__':  # pragma: no cover
    test = TestNsmrSimpleGymEnv()
    test.test_main()
    test.test_render()
    test.test_reward_params()
    test.test_set_config()