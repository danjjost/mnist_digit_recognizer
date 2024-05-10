from src.tictactoe.board import Board
from src.tictactoe.console_player import ConsolePlayer
from src.tictactoe.player import Player


class Game():
    def __init__(self):
        self.mute = False
        
        
    def initialize(self, playerX: Player, playerO: Player):
        self.current_player = playerX
        
        self.playerX = playerX
        self.playerO = playerO
        
        self.board = Board()
        
        
    def mute_game(self):
        self.mute = True
        
        
    def play(self, playerX: Player, playerO: Player):
        self.initialize(playerX, playerO)
        
        while True:
            self.print_board()
            self.get_move()
            if self.board.is_game_over():
                break
            
            self.switch_player()
            
        self.print_board()
        if self.mute == False:
            print(f'Player {self.current_player.symbol} wins!')
        
        return self.current_player
        
        
    def print_board(self):
        if self.mute == True:
            return
        
        print(f'{self.board.current[0]} | {self.board.current[1]} | {self.board.current[2]}')
        print('---------')
        print(f'{self.board.current[3]} | {self.board.current[4]} | {self.board.current[5]}')
        print('---------')
        print(f'{self.board.current[6]} | {self.board.current[7]} | {self.board.current[8]}')
        print('\n')
        
        
    def get_move(self):
        while True:
            move = self.current_player.get_move(self.board)
            try:
                self.board.move(move, self.current_player.symbol)
                break
            except ValueError:
                print('Invalid move, try again')
    
    
    def switch_player(self):
        if self.current_player.guid == self.playerX.guid:
            self.current_player = self.playerO
        else:
            self.current_player = self.playerX 
    

if __name__ == '__main__':
    game = Game(ConsolePlayer('X'), ConsolePlayer('O'), Board())
    winner = game.play()
    print(winner.guid)