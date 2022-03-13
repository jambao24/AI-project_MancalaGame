import unittest
import numpy as np
from maxPlayer import MaxPlayer

class TestMancalaBoard(unittest.TestCase):
    
  def setUp(self):    
    self.player = MaxPlayer(True)

  def test_getMoveGoAgain(self):
    board = np.array([4,4,4,4,4,1,15,4,4,4,4,4,4,10])
    self.assertEqual(self.player.getNextMove1(board), 5)
    board = np.array([4,4,4,4,4,4,15,4,4,4,4,4,4,10])
    self.assertEqual(self.player.getNextMove1(board), 2)
    board = np.array([4,4,4,4,4,14,15,4,4,4,4,4,4,10])
    self.assertEqual(self.player.getNextMove1(board), 5)
    board = np.array([4,4,4,4,4,4,15,4,4,4,4,4,4,10])
    self.assertEqual(self.player.getNextMove2(board), 9)
    board = np.array([4,4,4,4,4,4,15,4,4,4,4,4,14,10])
    self.assertEqual(self.player.getNextMove2(board), 12)
    board = np.array([4,4,4,4,4,4,15,4,4,4,4,4,1,10])
    self.assertEqual(self.player.getNextMove2(board), 12)

  def test_getMoveDontGoAgain(self):
    board = np.array([1,0,0,18,0,0,15,4,4,4,4,0,4,10])
    self.assertEqual(self.player.getNextMove1(board), 3)
    board = np.array([7,4,0,0,0,0,15,4,4,4,4,4,4,10])
    self.assertEqual(self.player.getNextMove1(board), 1)
    board = np.array([4,4,0,4,0,4,15,1,0,0,18,0,0,10])
    self.assertEqual(self.player.getNextMove2(board), 10)
    board = np.array([4,4,4,4,4,4,15,7,4,0,0,0,0,10])
    self.assertEqual(self.player.getNextMove2(board), 8)