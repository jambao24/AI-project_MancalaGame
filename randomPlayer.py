import numpy as np

from mancala import Player

class RandomPlayer(Player):

  def getNextMove(self, boardState) -> int:
    if self.isPlayer1:
      return np.random.choice(np.nonzero(boardState[:6])[0])
    else:
      return np.random.choice(np.nonzero(boardState[7:13])[0]) + 7