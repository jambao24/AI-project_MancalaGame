'''
The following heuristics were borrowed from this paper:
https://digitalcommons.andrews.edu/cgi/viewcontent.cgi?article=1259&context=honors pg. 9-10
'''

class Heuristic:

  @staticmethod
  def evalBoard(boardState, isPlayer1):
    pass

# H1
class LeftPitHoardHeuristic(Heuristic):

  @staticmethod
  def evalBoard(boardState, isPlayer1):
    return boardState[0] if isPlayer1 else boardState[7]

# H2
class PitHoardHeuristic(Heuristic):

  @staticmethod
  def evalBoard(boardState, isPlayer1):
    return sum(boardState[0:6]) if isPlayer1 else sum(boardState[7:13])

# H4
class MaxStoreHeuristic(Heuristic):

  @staticmethod
  def evalBoard(boardState, isPlayer1):
    return boardState[6] if isPlayer1 else boardState[13]

