"""
    Apply Monte-Carlo control to Easy21. Initialise the value function to zero.

    Use a time-varying scalar step-size of alpha_t = 1/N(s_t; a_t)
    and an greedy exploration strategy with epsilon_t = N_0/(N_0 + N(s_t)), where N_0 = 100 is a constant,
    N(s) is the number of times that state s has been visited, and
    N(s; a) is the number of times that action a has been selected from state s.

    Feel free to choose an alternative value for N_0, if it helps producing better results.

    Plot the optimal value function V*(s) = max_a Q*(s; a) using similar axes to the following figure
    taken from Sutton and Barto's Blackjack example. Their policy was: stick if sum of cards >= 20.

    Note: Each state is defined by the dealer's initial score and the players sum during the game.
          The dealer can have any value of 1-10 initially. The player should have a score of 1-21.
          The value function now defines in each state "how well" the player is doing in the game.
    """
import math

import numpy as np
import collections
import pickle


class MonteCarloPolicyEvaluation(object):
    """
        A model-free (MDP transitions/rewards are unknown) RL approach using episode sampling (no bootstrapping).

        Monte-Carlo policy evaluation uses *empirical mean* return instead of expected return.

        Note: We use here observation and state interchangebly.

        Hint: This might be a sub-class of stable_baselines.common.base_class.BaseRLModel
    """

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.action_space = env.action_space
        # Hold estimated total return for state-action pairs
        self.Q = collections.defaultdict(dict)
        self.N_sa = collections.defaultdict(int)

    def learn(self, total_timesteps):
        """
            We sample at every episode from the game.

            Hint: This method might invoke a Runner(AbstractEnvRunner) to perform learning.

            :param total_timesteps:
            :return: the estimated value function
        """
        for episode in range(total_timesteps):
            if episode % 100_000 == 0:
                print("Episode: " + str(episode))
            episode_observations = collections.defaultdict(int)
            episode_observations_and_actions = collections.defaultdict(int)
            initial_observation = self.env.reset()
            episode_observations[initial_observation] += 1
            # We need the trajectory over the whole episode to count all the state visits.
            current_observation = initial_observation
            while True:  # for every time-step, do
                # self.env.render()
                selected_action = self.policy.step(current_observation, self.Q)
                episode_observations[current_observation] += 1
                episode_observations_and_actions[(current_observation, selected_action)] += 1
                current_observation, reward, done, _ = self.env.step(action=selected_action)
                if done:
                    # We ignore the ending scores from the dealer and player
                    break
            # Still we apply the same final episode reward to all observations in this episode
            self.__policy_update(episode_observations_and_actions, reward)
        return self

    def __policy_update(self, observations_and_actions, reward):
        for (obs, action) in observations_and_actions:
            # N(s,a) <- N(s,a) + 1
            self.N_sa[(obs, action)] += 1
            # alpha_t = 1/N(s_t; a_t)
            alpha = np.true_divide(1, self.N_sa[(obs, action)])
            # Q(s,a) <- Q(s,a) + alpha * [R - Q(s,a)]
            q = self.q_value(obs, action)
            self.Q[obs][action] += alpha * (reward - q)

    def q_value(self, observation, action):
        # We use a dict-to-list of actions for easier step-processing
        if action not in self.Q[observation]:
            self.Q[observation][action] = 0.0  # We initialize everything with zero
        return self.Q[observation][action]

    def q_value_max(self, observation):
        """
        :return: the max action-value for the given observation
        """
        if observation not in self.Q:
            return 0.0
        actions = self.Q[observation]
        max_value = -math.inf
        for action in actions:
            action_value = self.Q[observation][action]
            if action_value == 0.0:
                continue
            if action_value > max_value:
                max_value = action_value
        return max_value

    def q_max(self, observation):
        if observation not in self.Q:
            return -1
        actions = self.Q[observation]
        max_value = -math.inf
        max_action = -1
        for action in actions:
            action_value = self.Q[observation][action]
            if action_value == 0.0:
                continue
            if action_value > max_value:
                max_value = action_value
                max_action = action
        return max_action

    def q_values(self):
        by_action = collections.defaultdict(list)
        for obs, actions in self.Q.items():
            for action in actions:
                by_action[action].append(obs)
        for action in by_action:
            print("Action: " + str(action))
            for obs in by_action[action]:
                print(obs)
            print()

    def save(self, name):
        file_path = 'mc_%s.pkl' % name
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(name):
        file_path = 'mc_%s.pkl' % name
        with open(file_path, 'rb') as f:
            return pickle.load(f)
