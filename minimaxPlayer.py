import numpy as np
import math
from mancala import Player, MancalaBoard

class MinimaxPlayer(Player):

    def __init__(self, isPlayer1) -> None:
        super().__init__(isPlayer1)
        if isPlayer1:
            self.getNextMove = self.getNextMove1
        else:
            self.getNextMove = self.getNextMove2

    def getNextMove1(self, boardState) -> int:

        board = MancalaBoard()
        board.board = np.copy(boardState)
        bestMove = self.minimax1(board, 0, True)
        return bestMove[0]

    def getNextMove2(self, boardState) -> int:
        board = MancalaBoard()
        board.board = np.copy(boardState)
        bestMove = self.minimax2(board, 0, True)      
        return bestMove[0]

    # function -> minimax 
    #   returns a tuple (max pit index, max relative sum of pits)
    # parameters:
    # boardState -> board object
    # depth -> current depth (increasing)
    # player -> true (maximizer) | false (minimizer)
    # player1 -> true (player1) | false (player2)

    def minimax1(self, boardState, depth, player):
        #Player 1
        if depth == 4:
            maxStonesInPit = -math.inf
            indexResultingInMaxPit = -1
            stoneSet = []
            for x in range(0, boardState.P1_STORE):
            
                board = MancalaBoard()
                board.board = np.copy(boardState.board)
                board.playPit(x)
                if board.isGameOver():
                    return tuple([x, board.board[boardState.P1_STORE]])
                stoneSet.append(board.board[boardState.P1_STORE])
            return tuple([stoneSet.index(max(stoneSet)),max(stoneSet)])

        if player == True:
            minStonesInPit = math.inf
            indexResultingInMinPit = -1
            stoneSet = []
            for x in range(0, boardState.P1_STORE):
                board = MancalaBoard()
                board.board = np.copy(boardState.board)
                board.playPit(x)
                if board.isGameOver():
                    return tuple([x, board.board[boardState.P1_STORE]])
                stoneSet.append(board.board[boardState.P1_STORE])
                store = self.minimax1(board, depth + 1, False)[1]
                stoneSet[x] += store
            return tuple([stoneSet.index(max(stoneSet)),max(stoneSet)])
        else:
            minStonesInPit = math.inf
            indexResultingInMinPit = -1
            stoneSet = []
            for x in range(7, boardState.P2_STORE):
                board = MancalaBoard()
                board.board = np.copy(boardState.board)
                board.playPit(x)
                if board.isGameOver():
                    return tuple([x-7, board.board[boardState.P1_STORE]])
                stoneSet.append(board.board[boardState.P1_STORE])
                store = self.minimax1(board, depth + 1, True)[1]
                
                stoneSet[x-7] += store
            return tuple([stoneSet.index(min(stoneSet)),min(stoneSet)])

    def minimax2(self, boardState, depth, player):
        #Player 2
        if depth == 4:
            maxStonesInPit = -math.inf
            indexResultingInMaxPit = -1
            stoneSet = []
            for x in range(7, boardState.P2_STORE):
            
                board = MancalaBoard()
                board.board = np.copy(boardState.board)
                board.playPit(x)
                if board.isGameOver():
                    return tuple([x, board.board[boardState.P2_STORE]])
                stoneSet.append(board.board[boardState.P2_STORE])
            return tuple([stoneSet.index(max(stoneSet))+ 7,max(stoneSet)])

        if player == True:
            minStonesInPit = math.inf
            indexResultingInMinPit = -1
            stoneSet = []
            for x in range(7, boardState.P2_STORE):
            
                board = MancalaBoard()
                board.board = np.copy(boardState.board)
                board.playPit(x)
                if board.isGameOver():
                    return tuple([x, board.board[boardState.P2_STORE]])
                stoneSet.append(board.board[boardState.P2_STORE])
                store = self.minimax2(board, depth + 1, False)[1]
                stoneSet[x-7] += store
            return tuple([stoneSet.index(max(stoneSet)) + 7,max(stoneSet)])
        else:
            minStonesInPit = math.inf
            indexResultingInMinPit = -1
            stoneSet = []
            for x in range(7, boardState.P2_STORE):
                board = MancalaBoard()
                board.board = np.copy(boardState.board)
                board.playPit(x)
                if board.isGameOver():
                    return tuple([x, board.board[boardState.P2_STORE]])
                stoneSet.append(board.board[boardState.P2_STORE])
                store = self.minimax2(board, depth + 1, True)[1]
                stoneSet[x-7] += store
            return tuple([stoneSet.index(min(stoneSet)) + 7,min(stoneSet)])
