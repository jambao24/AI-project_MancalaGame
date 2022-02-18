from random import randrange

from mancala import Player

class RandomPlayer(Player):

  def getNextMove(self, boardState) -> int:
    return randrange(0,6) if self.isPlayer1 else randrange(7,13)