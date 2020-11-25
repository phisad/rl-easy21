import collections
from stable_baselines.common import policies
import numpy as np


class EpsilonGreedyActionPolicy(policies.BasePolicy):

    def __init__(self, observation_space, action_space):
        super(EpsilonGreedyActionPolicy, self).__init__(sess=None, ob_space=observation_space, ac_space=action_space,
                                                        n_env=1, n_steps=1, n_batch=1, reuse=False, scale=False)
        self.action_space = action_space
        self.action_space_range = np.arange(action_space.n)
        self.observation_space = observation_space
        # Hold estimated total return for state-action pairs
        self.Q = collections.defaultdict(dict)
        # Hold counts for state-action pairs
        self.N_sa = collections.defaultdict(int)
        # Hold counts for states
        self.N_s = collections.defaultdict(int)
        self.N_0 = 100

    def step(self, obs, state=None, mask=None):
        # estimated total returns by actions taken from this state
        state_actions = self.Q[obs]
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

    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError()


class Stick20ActionPolicy(policies.BasePolicy):

    def __init__(self, observation_space, action_space):
        super(Stick20ActionPolicy, self).__init__(sess=None, ob_space=observation_space, ac_space=action_space,
                                                  n_env=1, n_steps=1, n_batch=1, reuse=False, scale=False)
        self.action_space = action_space
        self.observation_space = observation_space

    def step(self, obs, state=None, mask=None):
        player_score = obs[1]
        if player_score >= 20:
            return 0  # STICK
        return 1  # HIT

    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError()

    def update(self, observations_and_actions, reward):
        pass


class RandomActionPolicy(policies.BasePolicy):

    def __init__(self, observation_space, action_space):
        super(RandomActionPolicy, self).__init__(sess=None, ob_space=observation_space, ac_space=action_space,
                                                 n_env=1, n_steps=1, n_batch=1, reuse=False, scale=False)
        self.action_space = action_space
        self.observation_space = observation_space

    def step(self, obs, state=None, mask=None):
        return self.action_space.sample()

    def proba_step(self, obs, state=None, mask=None):
        raise NotImplementedError()

    def update(self, observations_and_actions, reward):
        pass
