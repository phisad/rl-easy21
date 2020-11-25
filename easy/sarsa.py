"""
    Implement Sarsa(lambda) in 21s. Initialise the value function to zero. Use the same
    step-size and exploration schedules as in the previous section.

    > Use a time-varying scalar step-size of alpha_t = 1/N(s_t; a_t)
    > and an greedy exploration strategy with epsilon_t = N_0/(N_0 + N(s_t)), where N_0 = 100 is a constant,
    > N(s) is the number of times that state s has been visited, and
    > N(s; a) is the number of times that action a has been selected from state s.

    Run the algorithm with parameter values lambda in [0, 0.1, 0.2, ... 1].

    Stop each run after [Train for] 1000 episodes and report the mean-squared error sum_(s,a)[Q(s,a) - Q*(s,a)]^^2
    over all states s and actions a, comparing the true values Q*(s,a) computed in the previous section
    with the estimated values Q(s, a) computed by Sarsa. Plot the mean-squared error against lambda.
    (One comparison plot against Monte Carlo with MSE on y and Lambda on x with 10 data-points (one for each lambda))

    For lambda = 0 and lambda = 1 only, plot the learning curve of mean-squared error against episode number.
    (Two additional plots with MSE on y and episode on x with 1000 data points (one for each episode))

    Note: Each state is defined by the dealer's initial score and the players sum during the game.
          The dealer can have any value of 1-10 initially. The player should have a score of 1-21.
          The value function now defines in each state "how well" the player is doing in the game.

    """
import pickle

import numpy as np
import collections
import math

from easy.policies import EpsilonGreedyActionPolicy


class Sarsa(object):
    """
        Note: We use here observation and state interchangebly.

        Hint: This might be a sub-class of stable_baselines.common.base_class.BaseRLModel
    """

    def __init__(self, env, policy, _lambda):
        self.env = env
        self.policy = policy
        self._lambda = _lambda
        # Hold estimated total return for state-action pairs (should be a matrix S x A)
        self.Q = collections.defaultdict(dict)
        self.N_sa = collections.defaultdict(int)

    def learn(self, total_timesteps):
        """
            We sample at every time-step from the game.

            Hint: This method might invoke a Runner(AbstractEnvRunner) to perform learning.

            :param total_timesteps:
            :return: the estimated value function
        """
        for episode in range(total_timesteps):
            # Eligibility traces
            E = collections.defaultdict(int)
            previous_observation = self.env.reset()
            previous_action = self.policy.step(previous_observation, self.Q)
            self.N_sa[(previous_observation, previous_action)] += 1
            while True:  # for every time-step, do
                current_observation, reward, done, _ = self.env.step(action=previous_action)
                # Policy Update
                current_action = self.policy.step(current_observation, self.Q)
                self.N_sa[(current_observation, current_action)] += 1
                delta = reward + self.q_value(current_observation, current_action) - self.q_value(
                    previous_observation, previous_action)
                alpha = np.true_divide(1, self.N_sa[(current_observation, current_action)])  # alpha_t = 1/N(s_t; a_t)
                E[(previous_observation, previous_action)] += 1
                for s in self.Q:
                    for a in self.Q[s]:
                        q = self.q_value(s, a)  # creates the entry if not exist
                        self.Q[s][a] = q + alpha * delta * E[(s, a)]
                        E[(s, a)] *= self._lambda
                previous_action = current_action
                previous_observation = current_observation
                # Update after each timestep given an action, observation and reward
                if done:
                    break
        return self

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
        file_path = 'sarsa_%s.pkl' % name
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(name):
        file_path = 'sarsa_%s.pkl' % name
        with open(file_path, 'rb') as f:
            return pickle.load(f)
