import collections
import numpy as np


class EpsilonGreedyActionPolicy(object):
    """
        Might implement stable_baselines.common.policies.BasePolicy
    """

    def __init__(self, action_space):
        self.action_space = action_space
        self.action_space_range = np.arange(action_space.n)
        # Hold counts for states
        self.N_s = collections.defaultdict(int)
        self.N_0 = 100

    def step(self, obs, q=None):
        # estimated total returns by actions taken from this state
        state_actions = q[obs]
        self.N_s[obs] += 1
        if not state_actions:  # not seen before, all actions have equal change
            selected_action = np.random.choice(self.action_space_range)
        else:
            greedy_action = max(state_actions, key=state_actions.get)
            # epsilon slowly converges to zero for higher counts (starting from 1)
            epsilon = np.true_divide(self.N_0, self.N_0 + self.N_s[obs])
            # put equally low prob on all actions
            epsilon_greedy_probs = np.full(self.action_space.n, np.true_divide(epsilon, self.action_space.n))
            # put the highest prob on argmax (greedy) action
            epsilon_greedy_probs[greedy_action] += 1 - epsilon
            selected_action = np.random.choice(self.action_space_range, p=epsilon_greedy_probs)
        return selected_action


class Stick20ActionPolicy(object):
    """
        Might implement stable_baselines.common.policies.BasePolicy
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def step(self, obs, q=None):
        player_score = obs[1]
        if player_score >= 20:
            return 0  # STICK
        return 1  # HIT


class RandomActionPolicy(object):
    """
        Might implement stable_baselines.common.policies.BasePolicy
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def step(self, obs, q=None):
        return self.action_space.sample()
