import chess as ch
import random as rd
import chess.engine
import pandas as pd
import time as timer
import ChessNet as cn

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Engine:

    def __init__(self, board, maxDepth, color):
        self.board = board
        self.color = color
        self.maxDepth = maxDepth
        self.evalFunct = self.evalFunct
        self.neuralNet = cn.ChessNet(128)
        # Mapping from piece type to index in the board3d array
        self.piece_to_index = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }

    def aiEvalFunct(self):
        print("Board:", self.board.tensorBoard())
        output = self.neuralNet(self.board.tensorBoard())
        print("Output of the network:", output)
        return output
    
    def evalFunct(self):
        compt = 0
        #Sums up the material values
        for i in range(64):
            compt+=self.squareResPoints(ch.SQUARES[i])
        compt += self.mateOpportunity() + self.openning()
        return compt

    def mateOpportunity(self):
        if (self.board.legal_moves.count()==0):
            if (self.board.turn == self.color):
                return -999
            else:
                return 999
        else:
            return 0

    #to make the engine developp in the first moves
    def openning(self):
        if (self.board.fullmove_number<10):
            if (self.board.turn == self.color):
                return 1/30 * self.board.legal_moves.count()
            else:
                return -1/30 * self.board.legal_moves.count()
        else:
            return 0

    #Takes a square as input and 
    #returns the corresponding Hans Berliner's
    #system value of it's resident
    def squareResPoints(self, square):
        pieceValue = 0
        if(self.board.piece_type_at(square) == ch.PAWN):
            pieceValue = 1
        elif (self.board.piece_type_at(square) == ch.ROOK):
            pieceValue = 5.1
        elif (self.board.piece_type_at(square) == ch.BISHOP):
            pieceValue = 3.33
        elif (self.board.piece_type_at(square) == ch.KNIGHT):
            pieceValue = 3.2
        elif (self.board.piece_type_at(square) == ch.QUEEN):
            pieceValue = 8.8

        if (self.board.color_at(square)!=self.color):
            return -pieceValue
        else:
            return pieceValue

        
    def getBestMove(self):
        # Always initialize move to a valid chess.Move object or None
        best_move, movecount = self.get_ai_move(self.board, self.maxDepth)
        return best_move

    def engine(self, candidate, depth):

        # Get list of legal moves of the current position
        move_list = list(self.board.legal_moves)
        best_candidate = None
        best_move = None  # To track the best move
        
        # If maxDepth is 1, return a random move
        if self.maxDepth == 1:
            return move_list[rd.randint(0, len(move_list) - 1)]
        
        # Reached max depth of search or no possible moves
        if depth == self.maxDepth or self.board.legal_moves.count() == 0:
            return self.evalFunct()

        # Initialize best_candidate for max or min
        if depth % 2 != 0:  # Engine's turn, maximize
            best_candidate = float("-inf")
        else:  # Human's turn, minimize
            best_candidate = float("inf")

        # Analyze board after deeper moves
        for move in move_list:
            self.board.push(move)
            value = self.engine(best_candidate, depth + 1)
            self.board.pop()

            # Basic minimax algorithm
            if depth % 2 != 0 and value > best_candidate:  # Maximizing
                best_candidate = value
                best_move = move
            elif depth % 2 == 0 and value < best_candidate:  # Minimizing
                best_candidate = value
                best_move = move

            # Alpha-beta pruning cuts
            if (candidate is not None and
                ((value < candidate and depth % 2 == 0) or
                 (value > candidate and depth % 2 != 0))):
                break

        return best_move if depth == 1 else best_candidate
    
    # Prediction function
    def predict(self, model, board3d):
        # Ensure the input has the correct shape and add a batch dimension
        board_tensor = torch.tensor(board3d, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            prediction = model(board_tensor)

        return prediction.item()

    #Pytorch optimized minimax
    def minimax_eval_ai(self, board):
        #start = timer.time()
        board3d = self.split_dims(board)
        #splittime = timer.time() - start
        board3d = np.expand_dims(board3d, 0)
        #expand_dims_time = timer.time() - splittime - start
        pred = self.predict(board3d)
        #pred_time = timer.time() - splittime - expand_dims_time - start

        #print(f"Split time in seconds: {splittime:.8f},  expand dims time in seconds: {expand_dims_time:.8f}, pred time in seconds: {pred_time:.8f}")
        return pred
    
    #Pytorch optimized minimax
    def minimax_eval(self, board):
        return self.evalFunct()

    def minimax(self, board, depth, alpha, beta, maximizing_player, i):
        if depth == 0 or board.is_game_over():
            eval = self.minimax_eval(board)
            return eval, i+1

        if maximizing_player:
            max_eval = -np.inf
            for move in board.legal_moves:
                board.push(move)
                eval, i = self.minimax(board, depth - 1, alpha, beta, False, i)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, i+1
        else:
            min_eval = np.inf
            for move in board.legal_moves:
                board.push(move)
                eval, i = self.minimax(board, depth - 1, alpha, beta, True, i)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, i+1

    def get_ai_move(self, board, depth):
        max_move = None
        max_eval = -np.inf

        i = 0
        for move in board.legal_moves:
            board.push(move)
            eval, i = self.minimax(board, depth - 1, -np.inf, np.inf, False, i)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                max_move = move

        return max_move, i
    
    def square_to_indices(self, square):
        return 7 - chess.square_rank(square), chess.square_file(square)

    # Optimized `split_dims` function
    def split_dims(self, board):
        board3d = np.zeros((14, 8, 8), dtype=np.int8)

        # Populate board3d for white pieces
        for piece_type, index in self.piece_to_index.items():
            squares = np.array(list(board.pieces(piece_type, chess.WHITE)))
            if squares.size:
                rows, cols = 7 - squares // 8, squares % 8
                board3d[index, rows, cols] = 1

        # Populate board3d for black pieces
        for piece_type, index in self.piece_to_index.items():
            squares = np.array(list(board.pieces(piece_type, chess.BLACK)))
            if squares.size:
                rows, cols = 7 - squares // 8, squares % 8
                board3d[index + 6, rows, cols] = 1

        # Add attacks/moves layers
        aux = board.turn
        board.turn = chess.WHITE
        for move in board.legal_moves:
            row, col = self.square_to_indices(move.to_square)
            board3d[12, row, col] = 1
        board.turn = chess.BLACK
        for move in board.legal_moves:
            row, col = self.quare_to_indices(move.to_square)
            board3d[13, row, col] = 1
        board.turn = aux

        return board3d

    def fen_to_board(self, fen):
        fen_board = chess.Board(fen)
        return fen_board

    def fen_to_board3d_array(self, fens):
        board3d_array = np.zeros(shape=(fens.shape[0], 14, 8, 8))
        for i in range(fens.shape[0]):
            #print(fen)
            fen_board = self.fen_to_board(fens[i])
            board_split = self.split_dims(fen_board)
            #print(board_split.shape)
            board3d_array[i] = board_split

        return board3d_array


    def get_fen_dataset(self, path):
        df = pd.read_csv(path)
        df.head()
        return df
