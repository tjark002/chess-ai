import chess as ch

class Engine: 

    def __init__(self, board, maxDepth, color): 
        self.board = board
        self.maxDepth = maxDepth
        self.color = color


    def engine(self, candidate, depth): 
        if (depth == self.maxDepth or self.board.legal_moves.count()==0):
            return self.evalFunct()
        
        else: 
            #get list of legal moves of the current postion
            moveList = list(self.board.legal_moves)

            #initialise newVandidate
            newCandidate = None

            if(depth % 2 != 0):
                newCandidate = float("-inf")
            else: 
                newCandidate = float("inf")

            for i in moveList:
                #Play the move i
                self.board.push(i)

                #Get the value of move i
                value = self.engine(newCandidate, depth+1)

                #Basic minmax algorithm:
                #if minimizing (engine's turn)
                

