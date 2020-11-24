import unittest

from easy.monte import MonteCarloPolicyEvaluation, Stick20ActionPolicy, RandomActionPolicy, EpsilonGreedyActionPolicy
from easy.game import Easy21

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker


class MyTestCase(unittest.TestCase):

    def test_epsilon_greedy_policy(self):
        game = Easy21()
        policy = EpsilonGreedyActionPolicy(game.observation_space, game.action_space)
        self.__learn_and_plot(game, policy, time_steps=1_000_000)

    def test_epsilon_greedy_policy(self):
        game = Easy21()
        policy = EpsilonGreedyActionPolicy(game.observation_space, game.action_space)
        for time_steps in [1_000, 10_000, 100_000]:
            self.__learn_and_plot(game, policy, time_steps)

    def test_random_policy(self):
        game = Easy21()
        policy = RandomActionPolicy(game.observation_space, game.action_space)
        for time_steps in [1_000, 10_000, 100_000]:
            self.__learn_and_plot(game, policy, time_steps)

    def test_stick_20_policy(self):
        game = Easy21()
        policy = Stick20ActionPolicy(game.observation_space, game.action_space)
        for time_steps in [1_000, 10_000, 100_000]:
            self.__learn_and_plot(game, policy, time_steps)

    def __learn_and_plot(self, game, policy, time_steps):
        mc = MonteCarloPolicyEvaluation(env=game, policy=policy)
        mc.learn(total_timesteps=time_steps)
        # Values over all initial dealer scores and player scores
        # Given the policy, the value should be near zero everywhere except when the player has a score of 21
        dealer_scores = np.arange(1, 11, 1)  # 10
        player_scores = np.arange(10, 22, 1)  # 12
        V = np.zeros(shape=(len(dealer_scores), len(player_scores)))
        for d_idx, dealer_score in enumerate(dealer_scores):
            for p_idx, player_score in enumerate(player_scores):
                value = mc.action_probability(observation=(dealer_score, player_score))
                V[d_idx][p_idx] = value

        D, P = np.meshgrid(dealer_scores, player_scores)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(D, P, V.transpose())  # somehow we have to transpose here
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_zlabel("Value")
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=1.0))
        plt.show()


if __name__ == '__main__':
    unittest.main()
