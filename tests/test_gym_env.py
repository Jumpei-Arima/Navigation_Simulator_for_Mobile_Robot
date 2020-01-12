from unittest import TestCase
import gym
import nsmr
print(__file__)

class TestNsmrGymEnv(TestCase):
    def test_main1(self):
        env = gym.make("nsmr-v0")
        obs = env.reset()

        # Try stepping a few times
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

if __name__ == '__main__':  # pragma: no cover
    test = TestNsmrGymEnv()
    test.test_main1()