import ChessEngine as ce
import chess as ch

class Main:

    def __init__(self, board=ch.Board):
        self.board=board

    #play human move
    def playHumanMove(self):
        try:
            print(self.board.legal_moves)
            print("""To undo your last move, type "undo".""")
            #get human move
            play = input("Your move: ")
            if (play=="undo"):
                self.board.pop()
                self.board.pop()
                self.playHumanMove()
                return
            self.board.push_san(play)
        except:
            self.playHumanMove()

    #play engine move
    def playEngineMove(self, maxDepth, color):
        engine = ce.Engine(self.board, maxDepth, color)
        self.board.push(engine.getBestMove())

    #start a game
    def startGame(self):
        #get the mode (p2e, e2e, p2p)
        mode=None
        while(mode!="p2e" and mode!="e2e" and mode!="p2p"):
            mode = input("""Choose mode (type "p2e", "e2e", or "p2p"): """)
        if mode=="p2e":
            self.p2e()
        elif mode=="e2e":
            self.e2e()
        elif mode=="p2p":
            self.p2p()
        


        #reset the board
        #self.board.reset
        #start another game
        #self.startGame()

    #play player vs player
    def p2p(self):
        #get first player's color
        color=None
        while(color!="b" and color!="w"):
            color = input("""First player (type "b" or "w"): """)
        while (self.terminate_game()==False):
            print(self.board)
            self.playHumanMove()
            print(self.board)
            if self.board.is_checkmate()==True:
                break
            self.playHumanMove()
        print(self.board)
        print(self.board.outcome())

    #play engine vs engine
    def e2e(self):
        maxDepth=None
        while(isinstance(maxDepth, int)==False):
            maxDepth = int(input("""Choose depth: """))
        while (self.terminate_game()==False):
            #print("The engine is thinking...")
            self.playEngineMove(maxDepth, ch.WHITE)
            #print(self.board)
            if self.terminate_game()==True:
                break
            #print("The engine is thinking...")
            self.playEngineMove(maxDepth, ch.BLACK)
            #print(self.board)
        print("The game is over!")
        print(self.board)
        print(self.board.outcome())

    def p2e(self):
        #get human player's color
        color=None
        while(color!="b" and color!="w"):
            color = input("""Play as (type "b" or "w"): """)
        maxDepth=None
        while(isinstance(maxDepth, int)==False):
            maxDepth = int(input("""Choose depth: """))
        if color=="b":
            while (self.terminate_game()==False):
                print("The engine is thinking...")
                self.playEngineMove(maxDepth, ch.WHITE)
                print(self.board)
                if self.terminate_game()==True:
                    break
                self.playHumanMove()
                print(self.board)
            print(self.board)
            print(self.board.outcome())    
        elif color=="w":
            while (self.terminate_game()==False):
                print(self.board)
                self.playHumanMove()
                print(self.board)
                if self.terminate_game()==True:
                    break
                print("The engine is thinking...")
                self.playEngineMove(maxDepth, ch.BLACK)
            print(self.board)
            print(self.board.outcome())


    def terminate_game(self):
        if self.board.is_checkmate()==True:
            return True
        elif self.board.is_stalemate()==True:
            return True
        elif self.board.is_insufficient_material()==True:
            return True
        elif self.board.is_seventyfive_moves()==True:
            return True
        elif self.board.is_fivefold_repetition()==True:
            return True
        elif self.board.is_variant_draw()==True:
            return True
        else:
            return False

#create an instance and start a game
newBoard= ch.Board()
game = Main(newBoard)
bruh = game.startGame()