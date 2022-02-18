from mancala import MancalaGame
from randomPlayer import RandomPlayer

def main():
  player1 = RandomPlayer(True)
  player2 = RandomPlayer(False)
  result = MancalaGame(player1,player2).run()
  if result == 1:
    print("Player 1 Won!")
  elif result == 0:
    print("Tie Game!")
  else:
    print("Player 2 Won!")

main()