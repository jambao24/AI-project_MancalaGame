#from fcntl import F_SEAL_SEAL
import numpy as np
from randomPlayer import RandomPlayer
from mancala import Player, MancalaBoard

class monteCarloPlayer(Player):

  def __init__(self, isPlayer1) -> None:
    super().__init__(isPlayer1)
    if isPlayer1:
      self.getNextMove = self.getNextMove1
    else:
      self.getNextMove = self.getNextMove2
    self.depth = 5
    self.sample_s = 300

  def getNextMove1(self, boardState) -> int:
    playsToEvalStones = [0]*14
    playsToEvalPlays = [0]*14
    playsToEvalAvgs = [0.0]*14
    for _ in range(0, self.sample_s):
      player1 = RandomPlayer(True)
      player2 = RandomPlayer(False)
      board = MancalaBoard()
      board.board = np.copy(boardState)
      d = 0
      isFirst = True
      isPlayer1Turn = True
      while(d < self.depth and not board.isGameOver()):
        if isPlayer1Turn:
          currPlayer = player1
          d+=1
        else:
          currPlayer = player2
        nextMove = currPlayer.getNextMove(board.board)
        if isFirst: 
          isFirst = False
          firstMove = nextMove
        if nextMove == 6 or nextMove == 13:
          print(f'out of bounds')
        if not board.playPit(nextMove):
          isPlayer1Turn = not isPlayer1Turn
      if board.isGameOver():
        board.collectRemaining()
      playsToEvalStones[firstMove]+= (board.board[6]-board.board[13])
      playsToEvalPlays[firstMove]+=1
    for r in range(0,14): 
      if playsToEvalPlays[r]>0:
        playsToEvalAvgs[r] = playsToEvalStones[r]/playsToEvalPlays[r]
    movement = np.where(playsToEvalAvgs == np.amax(playsToEvalAvgs))

    return movement[0][0]
    
  def getNextMove2(self, boardState) -> int:
    playsToEvalStones = [0]*14
    playsToEvalPlays = [0]*14
    playsToEvalAvgs = [0.0]*14
    for _ in self.sample_s:
      player1 = RandomPlayer(True)
      player2 = RandomPlayer(False)
      board = MancalaBoard()
      board.board = np.copy(boardState)
      d = 0
      isFirst = True
      isPlayer1Turn = False
      while(d < self.depth and not board.isGameOver()):
        if isPlayer1Turn:
          currPlayer = player1
        else:
          d+=1
          currPlayer = player2
        nextMove = currPlayer.getNextMove(board.board)
        if isFirst: 
          isFirst = False
          firstMove = nextMove
        if nextMove == 6 or nextMove == 13:
          print(f'out of bounds')
        if not board.playPit(nextMove):
          isPlayer1Turn = not isPlayer1Turn
      if board.isGameOver():
        board.collectRemaining()
      playsToEvalStones[firstMove]+=(board.board[13]-board.board[6])
      playsToEvalPlays[firstMove]+=1
    for r in range(0,14): 
      if playsToEvalPlays[r] > 0:
        playsToEvalAvgs[r] = playsToEvalStones[r]/playsToEvalPlays[r]
    movement = np.where(playsToEvalAvgs == np.amax(playsToEvalAvgs))

    return movement[0][0]