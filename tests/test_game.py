import unittest

from easy.game import Easy21


class MyTestCase(unittest.TestCase):

    def test_game(self):
        env = Easy21()
        env.reset()
        while True:
            env.render()
            (dealer, player), reward, done, _ = env.step(action=env.action_space.sample())
            if done:
                break
        env.render()
        self.assertIn(reward, [-1, 0, 1])
        self.__assert_scores(dealer, player, reward)

    def test_game_reset(self):
        game = Easy21()
        self.assertEqual(game.score_dealer, 0)
        self.assertEqual(game.score_player, 0)
        game.reset()
        self.assertNotEqual(game.score_dealer, 0)
        self.assertNotEqual(game.score_player, 0)

    def test_game_step_stick(self):
        game = Easy21()
        game.reset()
        (dealer, player), reward, done, _ = game.step(action=0)
        self.assertEqual(done, True, msg="Done should be 'True'")
        self.assertIn(reward, [-1, 0, 1], msg="Rewards should be in [-1, 0, 1]")
        self.__assert_scores(dealer, player, reward)

    def test_game_step_hit(self):
        game = Easy21()
        game.reset()
        dealer_before = game.score_dealer
        player_before = game.score_player
        (dealer, player), reward, done, _ = game.step(action=1)
        self.assertEqual(done, False)
        self.assertEqual(reward, 0)
        self.assertEqual(dealer_before, dealer)
        self.assertNotEqual(player_before, player)

    def __assert_scores(self, dealer, player, reward):
        if dealer > 21:  # busted
            self.assertTrue(reward == 1, msg="Player should have won with scores: " + str((dealer, player)))
        elif dealer > player:
            self.assertTrue(reward == -1, msg="Player should have lost with scores: " + str((dealer, player)))
        if player > 21:  # busted
            self.assertTrue(reward == -1, msg="Player should have lost with scores: " + str((dealer, player)))
        elif player > dealer:
            self.assertTrue(reward == 1, msg="Player should have won with scores: " + str((dealer, player)))
        if player == dealer:
            self.assertTrue(reward == 0, msg="Should be a draw: " + str((dealer, player)))

    if __name__ == '__main__':
        unittest.main()
