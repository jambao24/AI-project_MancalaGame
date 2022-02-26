'''
Some of the following heuristics were borrowed from this paper:
https://digitalcommons.andrews.edu/cgi/viewcontent.cgi?article=1259&context=honors pg. 9-10
'''

class Heuristic:
  @staticmethod
  def evalBoard(boardState, isPlayer1):
    pass


class LeftPitHoardHeuristic(Heuristic):
  @staticmethod
  def evalBoard(boardState, isPlayer1):
    return sum(boardState[0:3]) if isPlayer1 else sum(boardState[7:10])


class RightPitHoardHeuristic(Heuristic):
  @staticmethod
  def evalBoard(boardState, isPlayer1):
    return sum(boardState[3:6]) if isPlayer1 else sum(boardState[10:13])


class PitHoardHeuristic(Heuristic):
  @staticmethod
  def evalBoard(boardState, isPlayer1):
    return sum(boardState[0:6]) if isPlayer1 else sum(boardState[7:13])


class MaxStoreHeuristic(Heuristic):
  @staticmethod
  def evalBoard(boardState, isPlayer1):
    return boardState[6] if isPlayer1 else boardState[13]


class StoreDifferenceHeuristic(Heuristic):
  @staticmethod
  def evalBoard(boardState, isPlayer1):
    return boardState[6] - boardState[13] if isPlayer1 else boardState[13] - boardState[6]

