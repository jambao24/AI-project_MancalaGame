#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 12:32:53 2022

@author: vickyhaney
"""
import numpy as np

from mancala import Player, MancalaBoard


class MaxPlayer(Player):
    
    def __init__(self, isPlayer1) -> None:  #gets called when you create player
      super().__init__(isPlayer1)
      if isPlayer1:
        self.getNextMove = self.getNextMove1
      else:
        self.getNextMove = self.getNextMove2
        
    def getNextMove1(self, boardState) -> int:
      playableMoves = np.nonzero(boardState[:6])[0]
      for k in np.flip(playableMoves):
        if ((boardState[k] % 13) + k == 6):
          return k

      scores = []
      game = MancalaBoard()
      for k in playableMoves:
        board = np.copy(boardState)
        game.board = board
        game.playPit(k)
        scores.append(game.board[6])
      maxScore = np.amax(scores)
      indices = np.argwhere(scores==maxScore).flatten()
      return playableMoves[np.random.choice(indices)]
      
    def getNextMove2(self, boardState) -> int:
      playableMoves = np.nonzero(boardState[7:13])[0] + 7
      for k in np.flip(playableMoves):
        if ((boardState[k] % 13) + k == 13):
          return k

      scores = []
      game = MancalaBoard()
      for k in playableMoves:
        board = np.copy(boardState)
        game.board = board
        game.playPit(k)
        scores.append(game.board[13])
      maxScore = np.amax(scores)
      indices = np.argwhere(scores==maxScore).flatten()
      return playableMoves[np.random.choice(indices)]
            
            
           
        
        
