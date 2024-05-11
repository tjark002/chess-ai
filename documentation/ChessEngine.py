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

from joblib import Parallel, delayed
from collections import defaultdict

class Engine:

    def __init__(self, board, maxDepth, model=None):
        self.board = board
        self.maxDepth = maxDepth
        self.evalFunct = self.evalFunct
        self.model = model
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
    
    def evalFunct(self, color):
        compt = 0
        #Sums up the material values
        for i in range(64):
            compt+=self.squareResPoints(ch.SQUARES[i], color)
        compt += self.mateOpportunity(color) + self.openning(color)
        return compt

    def mateOpportunity(self, color):
        if (self.board.legal_moves.count()==0):
            if (self.board.turn == color):
                return -999
            else:
                return 999
        else:
            return 0

    #to make the engine developp in the first moves
    def openning(self, color):
        if (self.board.fullmove_number<10):
            if (self.board.turn == color):
                return 1/30 * self.board.legal_moves.count()
            else:
                return -1/30 * self.board.legal_moves.count()
        else:
            return 0

    #Takes a square as input and 
    #returns the corresponding Hans Berliner's
    #system value of it's resident
    def squareResPoints(self, square, color):
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

        if (self.board.color_at(square)!=color):
            return -pieceValue
        else:
            return pieceValue

        
    def getBestMove(self, color, ai_mode=True):
        board = self.board
        # Always initialize move to a valid chess.Move object or None
        max_move = None
        max_eval = -np.inf

        i = 0
        for move in board.legal_moves:
            board.push(move)
            eval, i = self.minimax(board, self.maxDepth - 1, -np.inf, np.inf, False, i, color, ai_mode)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                max_move = move

        return max_move, i
    
    # Prediction function
    def predict(self, board3d):
        # Ensure the input has the correct shape and add a batch dimension
        board_tensor = torch.tensor(board3d, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            prediction = self.model(board_tensor)

        return prediction.item()

    #Pytorch optimized minimax
    def minimax_eval_ai(self, board, transposition_table=defaultdict(lambda: None)):
        fen = board.fen()
        cached_eval = transposition_table.get(fen)
        if cached_eval is not None:
            return cached_eval
        board3d = self.split_dims(board)
        board3d = np.expand_dims(board3d, 0)
        pred = self.predict(board3d)

        # Store the evaluation in the transposition table
        transposition_table[fen] = pred
        return pred
    
    #Pytorch optimized minimax
    def minimax_eval(self, board, color):
        return self.evalFunct(color)

    def minimax(self, board, depth, alpha, beta, maximizing_player, i, color, ai_mode=True, transposition_table=defaultdict(lambda: None)):
        if depth == 0 or board.is_game_over():
            if ai_mode:
                eval = self.minimax_eval_ai(board, transposition_table)
            else:
                eval = self.minimax_eval(board, color)
            return eval, i+1

        if maximizing_player:
            max_eval = -np.inf
            #for move in board.legal_moves:
            for move in self.move_ordering(board, maximizing_player):
                board.push(move)
                eval, i = self.minimax(board, depth - 1, alpha, beta, False, i, color, ai_mode)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, i+1
        else:
            min_eval = np.inf
            #for move in board.legal_moves:
            for move in self.move_ordering(board, maximizing_player):
                board.push(move)
                eval, i = self.minimax(board, depth - 1, alpha, beta, True, i, color, ai_mode)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, i+1
        
    # Principal Variation Search (PVS) with Alpha-Beta Pruning and Move Ordering
    def pvs(self, board, depth, alpha, beta, maximizing_player, i, color, ai_mode=True, transposition_table=defaultdict(lambda: None)):
        if depth == 0 or board.is_game_over():
            if ai_mode:
                eval = self.minimax_eval_ai(board, transposition_table)
            else:
                eval = self.minimax_eval(board, color)
            return eval, i+1

        legal_moves = self.move_ordering(board, maximizing_player)
        if not legal_moves:
            return (0 if board.is_check() else 0.5), i

        first_move = True
        best_value = -np.inf if maximizing_player else np.inf

        for move in legal_moves:
            board.push(move)
            if first_move:
                value, i = self.pvs(board, depth - 1, alpha, beta, not maximizing_player, i, color, ai_mode=True, transposition_table=defaultdict(lambda: None))
                first_move = False
            else:
                if maximizing_player:
                    value, i = self.pvs(board, depth - 1, alpha, alpha + 1, not maximizing_player, i, color, ai_mode=True, transposition_table=defaultdict(lambda: None))
                    if alpha < value < beta:
                        value, i = self.pvs(board, depth - 1, value, beta, not maximizing_player, i, color, ai_mode=True, transposition_table=defaultdict(lambda: None))
                else:
                    value, i = self.pvs(board, depth - 1, beta - 1, beta, not maximizing_player, i, color, ai_mode=True, transposition_table=defaultdict(lambda: None))
                    if alpha < value < beta:
                        value, i = self.pvs(board, depth - 1, alpha, value, not maximizing_player, i, color, ai_mode=True, transposition_table=defaultdict(lambda: None))
            board.pop()

            if maximizing_player:
                best_value = max(best_value, value)
                alpha = max(alpha, value)
            else:
                best_value = min(best_value, value)
                beta = min(beta, value)

            if beta <= alpha:
                break

        return best_value, i + 1


    def get_ai_move(self, board, depth, color):
        max_move = None
        max_eval = -np.inf
        # Optimized evaluation function with transposition table
        transposition_table = defaultdict(lambda: None)

        i = 0
        for move in board.legal_moves:
            board.push(move)
            eval, i = self.pvs(board, depth - 1, -np.inf, np.inf, False, i, color, True, transposition_table)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                max_move = move

        return max_move, i
    

    # Move ordering heuristic function
    def move_ordering(self, board, maximizing_player):
        # Assign a score to each move
        def move_score(move):
            # Prioritize captures and checks
            score = 0
            if board.is_capture(move):
                score += 10
            if board.gives_check(move):
                score += 5
            return score

        moves = list(board.legal_moves)
        moves.sort(key=move_score, reverse=True)
        return moves








    
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
            row, col = self.square_to_indices(move.to_square)
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
    





    # Minimax function with Parallelization
    def pvs_parallel(self, board, depth, alpha, beta, maximizing_player, i, color, ai_mode=True, transposition_table=defaultdict(lambda: None)):
        results = Parallel(n_jobs=8, prefer="threads")(delayed(self.pvs_single)(board, move, depth, alpha, beta, maximizing_player, i, color, ai_mode, transposition_table) for move in board.legal_moves)
        best_result = max(results, key=lambda x: x[0])
        max_eval, max_move, total_nodes = best_result
        total_nodes = sum(x[2] for x in results)
        return max_move, total_nodes

    # Helper function for parallel processing
    def pvs_single(self, board, move, depth, alpha, beta, maximizing_player, i, color, ai_mode=True, transposition_table=defaultdict(lambda: None)):
        board.push(move)
        eval, nodes = self.pvs(board, depth - 1, alpha, beta, not maximizing_player, i, color, ai_mode, transposition_table)
        board.pop()
        return eval, move, nodes
    
    # Function to get the best move using Principal Variation Search
    def get_ai_move(self, board, depth, color, transposition_table=defaultdict(lambda: None)):
        max_move, total_nodes = self.pvs_parallel(board, depth, -np.inf, np.inf, True, 0, color, True, transposition_table)
        return max_move, total_nodes

############################################################################################################
# Legacy code
############################################################################################################
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
