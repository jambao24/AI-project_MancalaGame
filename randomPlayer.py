import numpy as np

from mancala import Player

class RandomPlayer(Player):

  def __init__(self, isPlayer1) -> None:
    super().__init__(isPlayer1)
    if isPlayer1:
      self.getNextMove = self.getNextMove1
    else:
      self.getNextMove = self.getNextMove2

  def getNextMove1(self, boardState) -> int:
    return np.random.choice(np.nonzero(boardState[:6])[0])
    
  def getNextMove2(self, boardState) -> int:
    return np.random.choice(np.nonzero(boardState[7:13])[0]) + 7