import chess as ch
import random as rd
import NeuralNet as nn

class Engine:

    def __init__(self, board, maxDepth, color):
        self.board = board
        self.color = color
        self.maxDepth = maxDepth
        self.evalFunct = self.aiEvalFunct
        self.neuralNet = nn.ChessNet()

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
        best_move = self.engine(None, 1)
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
