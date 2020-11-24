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

    def update(self, observations_and_actions, reward):
        for (obs, action) in observations_and_actions:
            # N(s,a) <- N(s,a) + 1
            self.N_sa[(obs, action)] += 1
            self.N_s[obs] += 1
            # Q(s,a) <- Q(s,a) + [G - Q(s,a)] / N(s,a)
            # We use a dict-to-list of actions for easier step-processing
            if action in self.Q[obs]:
                self.Q[obs][action] += np.true_divide(reward - self.Q[obs][action], self.N_sa[(obs, action)])
            else:
                self.Q[obs][action] = reward


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
        self.value_fn = collections.defaultdict(float)  # state -> estimated reward
        self.state_returns = collections.defaultdict(int)  # accumulate state reward
        self.state_visits = collections.defaultdict(int)  # accumulate state visits

    def learn(self, total_timesteps):
        """
            We sample at every episode from the game.

            What is the policy here though?

            Terms:
                G_t is the total reward over all previous timesteps (return).
                The value function v(s) is the 'expected' G_t (return) given a state (s) at t.
                Monte Carlo uses 'empirical mean' return instead of 'expected' return.

            Hint: This method might invoke a Runner(AbstractEnvRunner) to perform learning.

            :param total_timesteps:
            :return: the estimated value function
        """
        for episode in range(total_timesteps):
            episode_observations = collections.defaultdict(int)
            episode_observations_and_actions = collections.defaultdict(int)
            observation = self.env.reset()
            episode_observations[observation] += 1
            # We need the trajectory over the whole episode to count all the state visits.
            while True:  # for every time-step, do
                # self.env.render()
                selected_action = self.policy.step(observation)
                observation, reward, done, _ = self.env.step(action=selected_action)
                episode_observations[observation] += 1
                episode_observations_and_actions[(observation, selected_action)] += 1
                if done:
                    break
            # Still we apply the same final episode reward to all observations in this episode
            self.__incremental_update(episode_observations, reward)
            self.policy.update(episode_observations_and_actions, reward)
        return self

    def __incremental_update(self, epoch_observations, reward):
        """
            Updates the value function.
        """
        for state in epoch_observations:
            # Increment the total return after the episode for each visited state: S(s_t) = S(s_t) + G_t
            # where G_t is the total return over all episodes up to episode t.
            self.state_visits[state] += epoch_observations[state]
            self.state_returns[state] += reward
            # Update the value function for each visited state incrementally: V(s_t) = V(s_t) + alpha (G_t - V(s_t) )
            # where the update is the *weighted error term* between the true return and the estimated return.
            alpha = np.true_divide(1, self.state_visits[state])  # alpha_t = 1/N(s_t; a_t)
            self.value_fn[state] += alpha * (self.state_returns[state] - self.value_fn[state])

    def action_probability(self, observation):
        """
            We use this method here to return the estimated value instead of an action.
            Still, the action to be taken could base on the value-function given the observation.

            :param observation: a tuple of (dealer, player) score
            :return: the estimated value
        """
        if observation not in self.value_fn:
            return 0.0
        # The 'naive' value function would be the mean reward over all visits:
        # V(s) = S(s) / N(s) (total reward divided by total visits)
        return self.value_fn[observation]
