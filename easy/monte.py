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
import numpy as np
import collections


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
            self.__policy_update(episode_observations_and_actions, reward)
        return self

    def __policy_update(self, observations_and_actions, reward):
        for (obs, action) in observations_and_actions:
            # N(s,a) <- N(s,a) + 1
            self.policy.N_sa[(obs, action)] += 1
            self.policy.N_s[obs] += 1
            # Q(s,a) <- Q(s,a) + [G - Q(s,a)] / N(s,a)
            # We use a dict-to-list of actions for easier step-processing
            if action in self.policy.Q[obs]:
                self.policy.Q[obs][action] += np.true_divide(reward - self.policy.Q[obs][action],
                                                             self.policy.N_sa[(obs, action)])
            else:
                self.policy.Q[obs][action] = reward

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
