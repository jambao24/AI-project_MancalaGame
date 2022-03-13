#from deepQPlayer import DeepQPlayer
from mancala import MancalaGame
from maxPlayer import MaxPlayer
from randomPlayer import RandomPlayer
#from monteCarloPlayer import monteCarloPlayer
from minimaxPlayer import MinimaxPlayer

def main():
#  player1 = DeepQPlayer(True) 
#  player1 = monteCarloPlayer(True)
  player1 = MaxPlayer(False)
  player2 = RandomPlayer(True)
  #player1 = RandomPlayer(True)
#  player2 = monteCarloPlayer(False)
  #player2 = MinimaxPlayer(False)
  player1_wins = 0
  player2_wins = 0
  tie_games = 0
  for r in range(0, 1000):
    result = MancalaGame(player1, player2).run()
    if result == 1:
      player1_wins += 1
    elif result == -1:      
      player2_wins += 1
    else:
      tie_games += 1
    #print(f'r: {r}')
  print(f"Player 1 record: {player1_wins}-{player2_wins}-{tie_games} of games")

# def train():
#   DeepQPlayer(True).train()

main()
