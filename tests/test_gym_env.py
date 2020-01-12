from unittest import TestCase
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

    def test_randomize(self):
        env = NsmrGymEnv(randomize=True)
        obs = env.reset()

        # Try stepping a few times
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

    def test_render(self):
        env = NsmrGymEnv(randomize=True)
        obs = env.reset()

        # Try stepping a few times
        for i in range(10):
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

if __name__ == '__main__':  # pragma: no cover
    test = TestNsmrGymEnv()
    test.test_main()
    test.test_randomize()
    test.test_render()