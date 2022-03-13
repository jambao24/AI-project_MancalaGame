import itertools

from mancala import MancalaGame
from deepQPlayer import DeepQPlayer
from maxPlayer import MaxPlayer
from randomPlayer import RandomPlayer
from monteCarloPlayer import monteCarloPlayer
from minimaxPlayer import MinimaxPlayer


'''
main takes in 2 numbers as parameters to decide which mancala algorithm each player is playing as
a = Player 1, b = Player 2
0 = RandomPlayer
1 = MaxPlayer
2 = MiniMaxPlayer
3 = deepQPlayer
4 = monteCarloPlayer
'''
def main(a, b):
  if a == 0:
    player1 = RandomPlayer(True)
    p1 = "Random"
  elif a == 1:
    player1 = MaxPlayer(True)
    p1 = "Max"
  elif a == 2:
    player1 = MinimaxPlayer(True, 2)
    p1 = "Minimax"
  elif a == 3:
    player1 = DeepQPlayer(True)
    p1 = "DeepQ"
  elif a == 4:
    player1 = monteCarloPlayer(True)
    p1 = "monteCarlo"

  if b == 0:
    player2 = RandomPlayer(False)
    p2 = "Random"
  elif b == 1:
    player2 = MaxPlayer(False)
    p2 = "Max"
  elif b == 2:
    player2 = MinimaxPlayer(False, 2)
    p2 = "Minimax"
  elif b == 3:
    player2 = DeepQPlayer(False)
    p2 = "DeepQ"
  elif b == 4:
    player2 = monteCarloPlayer(False)
    p2 = "monteCarlo"

  player1_wins = 0
  player2_wins = 0
  tie_games = 0

  #for r in range(0, 1000):
  for r in range(0, 100):
    result = MancalaGame(player1, player2).run()
    if result == 1:
      player1_wins += 1
    elif result == -1:      
      player2_wins += 1
    else:
      tie_games += 1
    #print(f'r: {r}')
  print(f"{p1}-{p2} record: {player1_wins}-{player2_wins}-{tie_games} of games")

# def train():
#   DeepQPlayer(True).train()


'''run every possible combination of Player 1 vs Player 2 (5 different algorithms)'''
for i, j in itertools.product(range(5), range(5)):
  main(i, j)

#main(2,2)

'''
experimenting with different depth value pairs (1-1, 2-1, 3-2, 3-1, 4-4) 
'''
#result = MancalaGame(MinimaxPlayer(True, 4), MinimaxPlayer(False, 3)).run()
#print(result)
