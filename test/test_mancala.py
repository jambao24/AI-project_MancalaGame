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
    self.assertTrue((self.board.board == [0,5,5,5,0,5,7,5,0,4,4,4,4,0]).all())
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

  def test_lookahead(self):
    self.board.board = np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])
    lookahead1 = self.board.lookAhead(3)
    lookahead2 = self.board.lookAhead(8, lookahead1)
    self.assertTrue((lookahead1 == [4,4,4,0,5,5,1,5,4,4,4,4,4,0]).all())
    self.assertTrue((lookahead2 == [4,4,4,0,5,5,1,5,0,5,5,5,5,0]).all())
    self.assertTrue((self.board.board == [4,4,4,4,4,4,0,4,4,4,4,4,4,0]).all())
    self.assertEqual(sum(self.board.board), 48)

  def test_validMoves(self):
    self.board.board = np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])
    moves1 = self.board.getValidMoves(True)
    moves2 = self.board.getValidMoves(False)
    print(moves1)
    print(moves2)
    self.assertTrue(moves1 == [[0], [1], [2, 0],[2, 1],[2, 3],[2, 4],[2, 5],[3],[4],[5]])
    self.assertTrue(moves2 == [[7], [8], [9, 7],[9, 8],[9, 10],[9, 11],[9, 12],[10],[11],[12]])
    self.assertTrue((self.board.board == [4,4,4,4,4,4,0,4,4,4,4,4,4,0]).all())
    self.assertEqual(sum(self.board.board), 48)
