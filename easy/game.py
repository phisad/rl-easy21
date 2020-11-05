import gym
from gym import spaces
from gym.utils import seeding


class Easy21(gym.Env):
    """
    The game is played with an infinite deck of cards (i.e. cards are sampled with replacement).

    Each draw from the deck results in a value between 1 and 10 (uniformly distributed) with a colour of red
    (probability 1/3) or black (probability 2/3). There are no aces or picture (face) cards in this game.

    At the start of the game both the player and the dealer draw one black card (fully observed).

    Each turn the player may either stick or hit.
        If the player hits then she draws another card from the deck.
        If the player sticks she receives no further cards.

    The values of the player's cards are added (black cards) or subtracted (red cards)
        If the player's sum exceeds 21, or becomes less than 1, then she "goes bust" and loses the game (reward -1)

    If the player sticks then the dealer starts taking turns.

    The dealer always sticks on any sum of 17 or greater, and hits otherwise.

    If the dealer goes bust, then the player wins; otherwise, the outcome
    { win (reward +1), lose (reward -1), or draw (reward 0) } is the player with the largest sum.
    """

    def __init__(self):
        self.action_space = spaces.Discrete(2)  # (0) NOOP; (1) HIT; (2) STICK
        self.observation_space = spaces.MultiDiscrete([22, 22])  # only for documentation; can be negative; (0) NOOP
        self.reward_range = (-1, 1)
        self.random_value = None  # initialized in self.seed()
        self.seed()
        self.score_player = 0
        self.score_dealer = 0

    def seed(self, seed=None):
        self.random_value, seed = seeding.np_random(seed)
        return [seed]

    def scores(self):
        return self.score_dealer, self.score_player

    def __dealer_sticks(self):
        return self.score_dealer >= 17

    def __goes_bust(self, participant):
        if participant == "player":
            return self.score_player > 21 or self.score_player < 1
        elif participant == "dealer":
            return self.score_dealer > 21 or self.score_dealer < 1
        else:
            raise Exception("Unknown game participant: " + participant)

    def __take_card_by(self, participant, only_black=False):
        if participant == "player":
            self.score_player += self.__take_card(only_black)
        elif participant == "dealer":
            self.score_dealer += self.__take_card(only_black)
        else:
            raise Exception("Unknown game participant: " + participant)

    def __take_card(self, only_black=False):
        card_value = self.random_value.randint(11)  # high exclusive
        if only_black:
            return card_value
        if self.random_value.randint(101) <= 66:
            return card_value
        return -1 * card_value

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if action == 2:  # stick
            # perform dealer actions
            while not self.__dealer_sticks():
                self.__take_card_by("dealer")
                if self.__goes_bust("dealer"):
                    return self.scores(), +1, True, {}
            # game end: final comparison
            if self.score_player > self.score_dealer:
                return self.scores(), +1, True, {}
            if self.score_player == self.score_dealer:
                return self.scores(), 0, True, {}
            return self.scores(), -1, True, {}
        else:  # hit
            self.__take_card_by("player")
            if self.__goes_bust("player"):
                return self.scores(), -1, True, {}
        return self.scores(), 0, False, {}

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        self.score_dealer = 0
        self.score_player = 0
        self.__take_card_by("dealer", only_black=True)
        self.__take_card_by("player", only_black=True)
        return self.score_dealer, self.score_player

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        """
        print("Dealer: %s" % self.score_dealer)
        print("Player: %s" % self.score_player)
        print()
