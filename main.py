from deepQPlayer import DeepQPlayer
from mancala import MancalaGame
from randomPlayer import RandomPlayer

def main():
  player1 = DeepQPlayer(True)
  player2 = RandomPlayer(False)
  player1_wins = 0
  player2_wins = 0
  tie_games = 0
  for _ in range(1000):
    result = MancalaGame(player1, player2).run()
    if result == 1:
      player1_wins += 1
    elif result == -1:      
      player2_wins += 1
    else:
      tie_games += 1
  print(f"Player 1 record: {player1_wins}-{player2_wins}-{tie_games} of games")

def train():
  DeepQPlayer(True).train()

main()