import pickle
import unittest

from easy import game_plots
from easy.monte import MonteCarloPolicyEvaluation
from easy.policies import EpsilonGreedyActionPolicy
from easy.game import Easy21
from easy.sarsa import Sarsa
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

ACTION_STICK = 0
ACTION_HIT = 1

import numpy as np


class MyTestCase(unittest.TestCase):

    def test_epsilon_greedy__save(self):
        for _lambda in [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
            print("Training with lambda: %s" % _lambda)
            game = Easy21()
            policy = EpsilonGreedyActionPolicy(game.action_space)
            model = Sarsa(env=game, policy=policy, _lambda=_lambda)
            model.learn(total_timesteps=1000)
            model.save("greedy_1000_%s" % _lambda)

    def test_mse_save(self):
        mc = MonteCarloPolicyEvaluation.load("greedy1M")
        dealer_scores = np.arange(1, 11, 1)  # 10
        player_scores = np.arange(11, 22, 1)  # 11
        mse = dict()
        for _lambda in [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
            print("MSE for lambda: %s" % _lambda)
            sl = Sarsa.load("greedy_1000_%s" % _lambda)
            mse_value = 0
            counter = 0
            for action in [ACTION_STICK, ACTION_HIT]:
                for d in dealer_scores:
                    for p in player_scores:
                        mse_value += pow(sl.q_value((d, p), action) - mc.q_value((d, p), action), 2)
                        counter += 1
            mse["%s" % _lambda] = np.true_divide(mse_value, counter)
        MyTestCase.save(mse)

    def test_mse_plot(self):
        mse = MyTestCase.load()
        print(mse)
        x = np.arange(1.1, step=.1)
        y = []
        for _lambda in mse:
            y.append(mse[_lambda])
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o')
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=.1))
        plt.show()

    @staticmethod
    def save(obj):
        file_path = 'mse.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load():
        file_path = 'mse.pkl'
        with open(file_path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    unittest.main()
