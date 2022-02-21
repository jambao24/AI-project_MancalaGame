import unittest
import numpy as np
from mancala import MancalaBoard

class TestMancalaBoard(unittest.TestCase):
    
  def setUp(self):    
    self.board = MancalaBoard()

  def test_winning(self):
    self.board.board = np.array([4,4,4,4,4,4,15,4,4,4,4,4,4,10])
    self.assertEqual(self.board.isPlayer1Winning(), True)
    self.assertEqual(self.board.isGameTie(), False)
    self.board.board = np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,10])
    self.assertEqual(self.board.isPlayer1Winning(), False)
    self.assertEqual(self.board.isGameTie(), False)
    self.board.board = np.array([4,4,4,4,4,4,20,4,4,4,4,4,4,20])
    self.assertEqual(self.board.isPlayer1Winning(), False)
    self.assertEqual(self.board.isGameTie(), True)

  def test_play_pit(self):
    self.board.board = np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])
    go_again = self.board.playPit(3)
    self.assertTrue((self.board.board == [4,4,4,0,5,5,1,5,4,4,4,4,4,0]).all())
    self.assertEqual(sum(self.board.board), 48)
    self.assertFalse(go_again)

    self.board.board = np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])
    go_again = self.board.playPit(10)
    self.assertTrue((self.board.board == [5,4,4,4,4,4,0,4,4,4,0,5,5,1]).all())
    self.assertEqual(sum(self.board.board), 48)
    self.assertFalse(go_again)

    self.board.board = np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])
    go_again = self.board.playPit(2)
    self.assertTrue((self.board.board == [4,4,0,5,5,5,1,4,4,4,4,4,4,0]).all())
    self.assertEqual(sum(self.board.board), 48)
    self.assertTrue(go_again)

    self.board.board = np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])
    go_again = self.board.playPit(9)
    self.assertTrue((self.board.board == [4,4,4,4,4,4,0,4,4,0,5,5,5,1]).all())
    self.assertEqual(sum(self.board.board), 48)
    self.assertTrue(go_again)

  def test_capture_pit(self):
    self.board.board = np.array([4,4,4,4,0,5,1,5,5,4,4,4,4,0])
    go_again = self.board.playPit(0)
    self.assertTrue((self.board.board == [0,5,5,5,1,5,6,5,0,4,4,4,4,0]).all())
    self.assertEqual(sum(self.board.board), 48)
    self.assertFalse(go_again)

    self.board.board = np.array([4,4,4,4,4,4,0,4,4,0,5,5,5,1])
    go_again = self.board.playPit(5)
    self.assertTrue((self.board.board == [4,4,4,4,4,0,1,5,5,1,5,5,5,1]).all())
    self.assertEqual(sum(self.board.board), 48)
    self.assertFalse(go_again)

  def test_is_game_over(self):
    self.board.board = np.array([4,4,4,4,4,4,15,4,4,4,4,4,4,10])
    self.assertFalse(self.board.isGameOver())
    self.board.board = np.array([0,0,0,0,0,0,15,4,4,4,4,4,4,10])
    self.assertTrue(self.board.isGameOver())
    self.board.board = np.array([4,4,4,4,4,4,15,0,0,0,0,0,0,10])
    self.assertTrue(self.board.isGameOver())