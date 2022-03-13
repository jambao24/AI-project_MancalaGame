import numpy as np

class MancalaBoard:

  P1_STORE = 6
  P2_STORE = 13

  def __init__(self) -> None:
    # board is moving the stones clockwise during the game
    # indices 0-5 are player 1's pits
    # index 6 is player 1's store
    # indices 7-12 are player 2's pits
    # index 13 is player 2's store
    self.board = np.array([4,4,4,4,4,4,0,4,4,4,4,4,4,0])

  # returns true Go again
  def playPit(self, i) -> bool:
    assert i != self.P1_STORE and i != self.P2_STORE, "cannot play the store!"
    assert self.board[i] != 0, "move must have stones in pit!"
    isPlayer1Move = i < 6
    inHand = self.board[i]
    self.board[i] = 0
    while(inHand != 0):
      # skip the opponent's store
      if i == self.P2_STORE or (i == self.P2_STORE - 1 and isPlayer1Move):
        i = 0
      elif i == self.P1_STORE - 1 and not isPlayer1Move:
        i += 2
      else:
        i += 1
      self.board[i] += 1
      inHand -= 1
    # go again if last move is in player's store.
    if (isPlayer1Move and i == self.P1_STORE) or (not isPlayer1Move and i == self.P2_STORE):
      return True
    # capture pit if last move is in player's own empty pit.
    elif ((isPlayer1Move and i < 6) or (not isPlayer1Move and i > 6)) and self.board[i] == 1:
      self.capturePit(i)
    return False

  def playPits(self, moves):
    val = False
    for i in moves:
      val = self.playPit(i)
    return val

  # takes a move list so that you can formulate a lookahead 
  def lookAhead(self, move, board = None) -> np.array:
    cache = self.board.copy()
    self.board = self.board if board is None else board.copy()
    self.playPit(move)
    lookAheadBoard = self.board
    self.board = cache
    return lookAheadBoard

  def getValidMoves(self, isPlayer1, board = None):
    moves = []
    board = self.board if board is None else board
    nonZero = np.nonzero(board[:6])[0] if isPlayer1 else np.nonzero(board[7:13])[0] + 7
    store = self.P1_STORE if isPlayer1 else self.P2_STORE
    for i in nonZero:
      if (board[i] + i) % 14 == store:
        moves += list(map(lambda x: [i] + x, self.getValidMoves(isPlayer1, self.lookAhead(i,board))))
      else:
        moves.append([i])
    return moves

  def capturePit(self, i):
    if i < self.P1_STORE:
      self.board[self.P1_STORE] += self.board[i]
      self.board[self.P1_STORE] += self.board[12 - i]
    else:
      self.board[self.P2_STORE] += self.board[i]
      self.board[self.P2_STORE] += self.board[12 - i]
    self.board[i] = 0
    self.board[12 - i] = 0

  # at the end of the game you collect the remaining stones into the store.
  def collectRemaining(self):
    self.board[self.P1_STORE] += sum(self.board[0:6])
    self.board[self.P2_STORE] += sum(self.board[7:13])

  def isGameOver(self):
    return sum(self.board[0:6]) == 0 or sum(self.board[7:13]) == 0

  def isPlayer1Winning(self):
    return self.board[self.P1_STORE] > self.board[self.P2_STORE]

  def isGameTie(self):
    return self.board[self.P1_STORE] == self.board[self.P2_STORE]


class Player:

  def __init__(self, isPlayer1) -> None:
    self.isPlayer1 = isPlayer1

  def getNextMove(self, boardState: np.array) -> int:
    pass


class MancalaGame:

  def __init__(self, player1: Player, player2: Player) -> None:
    self.board = MancalaBoard()
    self.player1 = player1
    self.player2 = player2
    self.isPlayer1Turn = True

  # returns 1 if Player 1 won
  def run(self):
    while(not self.board.isGameOver()):
      if self.isPlayer1Turn:
        currPlayer = self.player1
      else:
        currPlayer = self.player2
#      currPlayer = self.player1 if self.isPlayer1Turn else self.player2
      nextMove = currPlayer.getNextMove(self.board.board)
      assert ((nextMove < 6 and currPlayer == self.player1) or 
       (nextMove > 6 and currPlayer == self.player2), "currPlayer {currPlayer}: {nextMove}")
      if nextMove == 6 or nextMove == 13:
          print(f'out of bounds1')
      if not self.board.playPit(nextMove):
        self.isPlayer1Turn = not self.isPlayer1Turn
    self.board.collectRemaining()
    if self.board.isPlayer1Winning():
      return 1
    elif self.board.isGameTie():
      return 0
    else:
      return -1



