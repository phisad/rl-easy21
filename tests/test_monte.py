import unittest

from easy.monte import MonteCarloPolicyEvaluation
from easy.policies import Stick20ActionPolicy, RandomActionPolicy, EpsilonGreedyActionPolicy
from easy.game import Easy21

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

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
        self.__plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_STICK))
        self.__plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_HIT))
        self.__plot(lambda d, p: mc.q_value_max(observation=(d, p)))
        self.__plot_image(lambda d, p: mc.q_max(observation=(d, p)))

    def test_epsilon_greedy_policy(self):
        game = Easy21()
        policy = EpsilonGreedyActionPolicy(game.action_space)
        for time_steps in [100_000]:
            mc = MonteCarloPolicyEvaluation(env=game, policy=policy)
            mc.learn(total_timesteps=time_steps)
            # self.__plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_STICK))
            # self.__plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_HIT))
            self.__plot(lambda d, p: mc.q_value_max(observation=(d, p)))
            self.__plot_image(lambda d, p: mc.q_max(observation=(d, p)))

    def test_random_policy(self):
        game = Easy21()
        policy = RandomActionPolicy(game.action_space)
        for time_steps in [1_000, 10_000, 100_000]:
            mc = MonteCarloPolicyEvaluation(env=game, policy=policy)
            mc.learn(total_timesteps=time_steps)
            self.__plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_STICK))
            self.__plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_HIT))
            self.__plot(lambda d, p: mc.q_value_max(observation=(d, p)))

    def test_stick_20_policy_save(self):
        game = Easy21()
        policy = Stick20ActionPolicy(game.action_space)
        for time_steps in [1_000]:
            mc = MonteCarloPolicyEvaluation(env=game, policy=policy)
            mc.learn(total_timesteps=time_steps)
        mc.save("stick20")

    def test_stick_20_policy_load(self):
        mc = MonteCarloPolicyEvaluation.load("stick20")
        # self.__plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_STICK))
        # self.__plot(lambda d, p: mc.q_value(observation=(d, p), action=ACTION_HIT))
        # self.__plot(lambda d, p: mc.q_value_max(observation=(d, p)))
        self.__plot_image(lambda d, p: mc.q_max(observation=(d, p)))

    def __plot_image(self, value_fn):
        # Values over all initial dealer scores and player scores
        # Given the policy, the value should be near zero everywhere except when the player has a score of 21
        dealer_scores = np.arange(1, 11, 1)  # 10
        player_scores = np.arange(11, 22, 1)  # 11
        V = np.zeros(shape=(len(dealer_scores), len(player_scores)))
        for d_idx, dealer_score in enumerate(dealer_scores):
            for p_idx, player_score in enumerate(player_scores):
                value = value_fn(dealer_score, player_score)
                V[d_idx][p_idx] = value
        fig, ax = plt.subplots()
        ax.imshow(V)
        ax.set_ylabel("Dealer initial showing")
        ax.yaxis.set_ticklabels(np.arange(0, 11, 1))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        ax.set_xlabel("Player sum")
        ax.xaxis.set_ticklabels(np.arange(10, 22, 1))
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        plt.show()

    def __plot(self, value_fn):
        # Values over all initial dealer scores and player scores
        # Given the policy, the value should be near zero everywhere except when the player has a score of 21
        dealer_scores = np.arange(1, 11, 1)  # 10
        player_scores = np.arange(11, 22, 1)  # 11
        V = np.zeros(shape=(len(dealer_scores), len(player_scores)))
        for d_idx, dealer_score in enumerate(dealer_scores):
            for p_idx, player_score in enumerate(player_scores):
                value = value_fn(dealer_score, player_score)
                V[d_idx][p_idx] = value

        D, P = np.meshgrid(dealer_scores, player_scores)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(D, P, V.transpose())  # somehow we have to transpose here
        ax.set_xlabel("Dealer initial showing")
        ax.set_ylabel("Player sum")
        ax.set_zlabel("Value")
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        plt.show()


if __name__ == '__main__':
    unittest.main()
