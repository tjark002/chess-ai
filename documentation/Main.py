import ChessEngine as ce
import chess as ch
import torch
from ChessNet import ChessNet
import time as timer

class Main:

    def __init__(self, board=ch.Board, model=None):
        self.board=board
        self.model=model

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
    def playEngineMove(self, maxDepth, color, engine=None, ai_mode=True):
        start = timer.time()
        best_move, move_count = engine.getBestMove(color, ai_mode)
        end = timer.time()
        elapsed_time = end - start
        if ai_mode:
            print(f"NEURAL NET: Movecount [{move_count}], Absolute time in seconds: {elapsed_time:.4f}, Moves per second: {move_count/elapsed_time:.2f}")
        else:
            print(f"BRUTEFORCE: Movecount [{move_count}], Absolute time in seconds: {elapsed_time:.4f}, Moves per second: {move_count/elapsed_time:.2f}")
        self.board.push(best_move)

        return move_count/elapsed_time

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
        engine = ce.Engine(self.board, maxDepth, self.model)
        i, j = 0, 0
        ai_kps, bf_kps = 0, 0
        while (self.terminate_game()==False):
            #print("AI: The engine is thinking...")
            ai_kps += self.playEngineMove(maxDepth, ch.WHITE, engine, True)
            i += 1
            #print(self.board)
            if self.terminate_game()==True:
                break
            #print("BRUTEFORCE: The engine is thinking...")
            bf_kps += self.playEngineMove(maxDepth, ch.BLACK, engine, False)
            j += 1
            #print(self.board)
        print("The game is over!")
        print(f"AI: {i} moves, {ai_kps/i:.2f} KPS")
        print(f"BRUTEFORCE: {j} moves, {bf_kps/j:.2f} KPS")
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
        engine = ce.Engine(self.board, maxDepth, self.model)
        ai_mode=None
        while(isinstance(ai_mode, bool)==False):
            ai_mode = bool(input("""Choose ai_mode: """))
        if color=="b":
            while (self.terminate_game()==False):
                print("The engine is thinking...")
                self.playEngineMove(maxDepth, ch.WHITE, engine)
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
                self.playEngineMove(maxDepth, ch.BLACK, engine)
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


# Load the model
model = ChessNet(hidden_size=128)
model.load_state_dict(torch.load('model.pth'))
print("Model loaded successfully!")
print(model)

#create an instance and start a game
newBoard= ch.Board()
game = Main(newBoard, model)



bruh = game.startGame()