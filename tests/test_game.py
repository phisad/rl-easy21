import unittest

from easy.game import Easy21


class MyTestCase(unittest.TestCase):

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
        (dealer, player), reward, done, _ = game.step(action=2)
        self.assertEqual(done, True)
        self.assertNotEqual(dealer, 0)
        self.assertNotEqual(player, 0)
        self.assertIn(reward, [-1, 0, 1])
        if reward == 1:
            self.assertTrue(player > dealer)
        if reward == 0:
            self.assertTrue(player == dealer)
        if reward == -1:
            self.assertTrue(player < dealer)

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

    if __name__ == '__main__':
        unittest.main()
