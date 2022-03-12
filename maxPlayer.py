#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 12:32:53 2022

@author: vickyhaney
"""
import numpy as np
#import math


from mancala import Player#, MancalaBoard
#from randomPlayer import RandomPlayer

class MaxPlayer(Player):
    
    def __init__(self, isPlayer1) -> None:  #gets called when you create player
      super().__init__(isPlayer1)
      if isPlayer1:
        self.getNextMove = self.getNextMove1
      else:
        self.getNextMove = self.getNextMove2
        
        
        

    def getNextMove1(self, boardState) -> int:

        list = []
        for k in (np.nonzero(boardState[:6])[0]):    #np.nonzero provides index of where there are pieces in which dish      
            #have the exact amount of pieces in a dish to end game in our score dish    
            if (k +  boardState[k] == (6-k)):#.all():             #we want closes well to bank
                list.append(k)
                #print(1)
                #print(list)
                #print(max(list))
                #print(boardState[max(list)])
                return max(list) #boardState[max(list)
         
            a = (np.mod(boardState[k],13) == 0)
            #have the exact amount of pieces that go around the game at least once, 
            #ends turn in scoring dish
            if (a + boardState[k] == (6-k)):#.all():        
            #we want closes well to bank
                list.append(k)
                #print(2)
                #print(list)
                #print(max(list))
                #print(boardState[max(list)])
                return max(list) #boardState[max(list)           
         
            
         
            #don't have exact amount of pieces but enough to get into score dish
            # and get on other side of the board
            elif (k +  boardState[k] > (6-k)):#.all():        #if index has more that 6 pieces, choose this one
                #print(k)
                list.append(k)
                #print(3)
                #print(list)
                #print(max(list))
                #print(boardState[max(list)])
                return max(list) #boardState[max(list)


            b = (np.mod(boardState[k],13) == 0)
            #have the exact amount of pieces that go around the game at least once, 
            #ends turn in scoring dish
            if (b + boardState[k] > (6-k)):#.all()        
            #we want closes well to bank
                list.append(k)
                #print(4)
                #print(list)
                #print(max(list))
                #print(boardState[max(list)])
                return max(list) #boardState[max(list)   


            
            else:
                return np.random.choice(np.nonzero(boardState[:6])[0])
                         

    #indexing nonzero
      
    def getNextMove2(self, boardState) -> int:
        list = []
        for j in (np.nonzero(boardState[7:13])[0]):
            if ((j+7) +  boardState[j+7] == ((6+7)-j)).all():#.all(1).any():                        #13  # (np.nonzero(boardState[:6])[0][j]+7)
                list.append(j)
                #print(list)
                return max(list) #boardState[max(list) 
           
            a = (np.mod(j,(13+7)) == 0)
            if (a +  boardState[j+7] == ((6+7)-j)).all:#.all(1).any():                        #13  # (np.nonzero(boardState[:6])[0][j]+7)
                list.append(j)
                #print(list)
                return max(list) #boardState[max(list)            
            
            #if index has more that 6 pieces, choose this one
            elif ((j+7) +  boardState[j+7] > ((6+7)-j)).all:#.all(1).any():                    #1    # +7 because of the indexing
                list.append(j)
                #print(list) 
                return  max(list) #boardState[max(list) 
             
            a = (np.mod(j,(13+7)) == 0)
            if (a +  boardState[j+7] > ((6+7)-j)).all:#.all(1).any():                        #13  # (np.nonzero(boardState[:6])[0][j]+7)
                list.append(j)
                #print(list)
                return max(list) #boardState[max(list)  


            else:
                return np.random.choice(np.nonzero(boardState[7:13])[0]) + 7
            
            
           
        
        
