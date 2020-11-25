import unittest

from easy import game_plots
from easy.monte import MonteCarloPolicyEvaluation
from easy.policies import Stick20ActionPolicy, RandomActionPolicy, EpsilonGreedyActionPolicy
from easy.game import Easy21

ACTION_STICK = 0
ACTION_HIT = 1


class MyTestCase(unittest.TestCase):

    def test_epsilon_greedy_policy_1M_save(self):
        game = Easy21()
        policy = EpsilonGreedyActionPolicy(game.action_space)
        mc = MonteCarloPolicyEvaluation(env=game, policy=policy)
        mc.learn(total_timesteps=1_000_000)
        mc.save("greedy1M")

    def test_epsilon_greedy_policy_1M_load(self):
        mc = MonteCarloPolicyEvaluation.load("greedy1M")
        game_plots.plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_STICK))
        game_plots.plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_HIT))
        game_plots.plot(lambda d, p: mc.q_value_max(observation=(d, p)))
        game_plots.plot_image(lambda d, p: mc.q_max(observation=(d, p)))

    def test_random_policy(self):
        game = Easy21()
        policy = RandomActionPolicy(game.action_space)
        for time_steps in [1_000, 10_000, 100_000]:
            mc = MonteCarloPolicyEvaluation(env=game, policy=policy)
            mc.learn(total_timesteps=time_steps)
            game_plots.plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_STICK))
            game_plots.plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_HIT))
            game_plots.plot(lambda d, p: mc.q_value_max(observation=(d, p)))
            game_plots.plot_image(lambda d, p: mc.q_max(observation=(d, p)))

    def test_stick_20_policy_save(self):
        game = Easy21()
        policy = Stick20ActionPolicy(game.action_space)
        for time_steps in [1_000]:
            mc = MonteCarloPolicyEvaluation(env=game, policy=policy)
            mc.learn(total_timesteps=time_steps)
        mc.save("stick20_%s" % time_steps)

    def test_stick_20_policy_load(self):
        mc = MonteCarloPolicyEvaluation.load("stick20_1000")
        game_plots.plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_STICK))
        game_plots.plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_HIT))
        game_plots.plot(lambda d, p: mc.q_value_max(observation=(d, p)))
        game_plots.plot_image(lambda d, p: mc.q_max(observation=(d, p)))


if __name__ == '__main__':
    unittest.main()
