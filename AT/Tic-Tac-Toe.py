## Importando as bibliotecas necessárias
import numpy as np
import random, copy


## Criando as classe de trabalho
class Game:
    def __init__(self, board=None):

        if board is None:
            self.board = np.zeros((3, 3), dtype=int)

        else:
            self.board = board

        self.game_over = False
        self.winner = 0
        self.score = 0

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.game_over = False
        self.winner = 0
        self.score = 0

    def play_move(self, move, player):

        if move is not None:
            board_flat = self.board.flatten()
            board_flat[move - 1] = player
            self.board = np.reshape(board_flat, (self.board.shape[0], self.board.shape[1]))

    def is_terminal(self):

        # verificando se o jogador 1 venceu
        if any(np.sum(self.board, axis=0) == 3) or \
                any(np.sum(self.board, axis=1) == 3) or \
                sum(np.diag(self.board)) == 3 or \
                sum(np.diag(self.board[::-1])) == 3:

            self.game_over = True
            self.winner = 1
            self.score = (len(self.get_availables()) + 1)

        # verificando se o jogador 2 venceu
        elif any(np.sum(self.board, axis=0) == -3) or \
                any(np.sum(self.board, axis=1) == -3) or \
                sum(np.diag(self.board)) == -3 or \
                sum(np.diag(self.board[::-1])) == -3:

            self.game_over = True
            self.winner = -1
            self.score = -(len(self.get_availables()) + 1)

        # verificando se houve empate
        else:
            if len(self.get_availables()) == 0:
                self.score = 0
                self.game_over = True
                self.winner = 0

            else:
                self.score = 0
                self.game_over = False
                self.winner = 0

        return self.game_over

    def get_availables(self):
        return np.where(self.board.flatten() == 0)[0] + 1

    def is_available(self, move):
        return self.board.flatten()[move - 1] == False

    def print_game(self):

        A = []

        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i, j] == -1:
                    A.append('O')
                elif self.board[i, j] == 1:
                    A.append('X')
                else:
                    A.append(' ')

        print(f' {A[0]} | {A[1]} | {A[2]}')
        print('---+---+---')
        print(f' {A[3]} | {A[4]} | {A[5]}')
        print('---+---+---')
        print(f' {A[6]} | {A[7]} | {A[8]}')
    #### Classe de algoritmos para o computador (I.A.)
class Computer:

    def __init__(self, algorithm='Random'):

        self.algorithm = algorithm.upper()

    def get_availables(self, board):
        return np.where(board.flatten() == 0)[0] + 1

    def is_terminal(self, board):

        score = 0
        game_over = 0
        winner = 0

        # verificando se o jogador 1 venceu
        if any(np.sum(board, axis=0) == 3) or any(np.sum(board, axis=1) == 3) or sum(np.diag(board)) == 3 or sum(np.diag(board[::-1])) == 3:

            game_over = True
            winner = 1
            score = (len(self.get_availables(board)) + 1)

        # verificando se o jogador 2 venceu
        elif any(np.sum(board, axis=0) == -3) or any(np.sum(board, axis=1) == -3) or sum(np.diag(board)) == -3 or sum(np.diag(board[::-1])) == -3:

            game_over = True
            winner = -1
            score = -(len(self.get_availables(board)) + 1)

        # verificando se houve empate
        else:
            if len(self.get_availables(board)) == 0:
                score = 0
                game_over = True
                winner = 0

            else:
                score = 0
                game_over = False
                winner = 0

        return game_over, score, winner

    def play_move(self, board, move, player):

        if move is not None:
            board_flat = board.flatten()
            board_flat[move - 1] = player
            board = np.reshape(board_flat, (board.shape[0], board.shape[1]))

        return board

    def play(self, board, player=-1, simulations=None):
        print('From class:\nComputer playing...')

        if self.algorithm == 'RANDOM':
            print('Random game')
            return self.random(board)

        if self.algorithm == 'MINIMAX':
            if player == 1:
                minimizing = False
            if player == -1:
                minimizing = True

            return self.minimax(board, minimizing=minimizing)

        if self.algorithm == 'ALPHA_BETA':
            return self.alpha_beta(board)

        if self.algorithm == 'MONTECARLO':
            return self.montecarlo(board, player=player, simulations=simulations)

    def random(self, board):

        # getting available moves
        return random.choice(self.get_availables(board))

    def minimax(self, board, minimizing=True):
        '''
         Receives the current board (state) and
         returns the selected move.
        '''

        if minimizing:
            print('Minimizando')
            ply = -1
            value, move = self.min_value(board, ply)

        else:
            print('Maximizando')
            ply = 1
            value, move = self.max_value(board, ply)

        return move

    def max_value(self, board, ply):
        # Verificando se é um estado terminal
        if self.is_terminal(board)[0]:
            return self.is_terminal(board)[1], None

        v = -np.inf

        avl = self.get_availables(board)
        for a in avl:
            new_board = self.play_move(board, a, ply)
            v2, _ = self.min_value(new_board, -ply)
            if v2 > v:
                v, move = v2, a

        return v, move

    def min_value(self, board, ply):
        # Verificando se é um estado terminal
        if self.is_terminal(board)[0]:
            return self.is_terminal(board)[1], None

        v = np.inf
        avl = self.get_availables(board)
        for a in avl:
            new_board = self.play_move(board, a, ply)
            v2, _ = self.max_value(new_board, -ply)

            if v2 < v:
                v, move = v2, a

        return v, move

    def alpha_beta(self, board):

        print('Returning a random choice')
        return self.random(board)

    def montecarlo(self, board, player=-1, simulations=None):

        # Definindo a última jogada
        if len(self.get_availables(board)) == 1:
            return self.random(board)

        # Definindo a quantidade de simulações
        if simulations is None:
            simulations = 1000

        # Vetor para salvar as jogadas
        win_moves = []

        # Realizando simulação de jogadas
        for i in range(simulations):

            # Inicializando a simulação
            ply = player
            local_board = copy.copy(board)

            # Selecionar o primeiro movimento utilizando a política aleatória
            move = self.random(local_board)

            # Expansão do estado
            local_board = self.play_move(local_board, move, ply)
            game_over, score, winner = self.is_terminal(local_board)

            # Realizando a simulação do jogo
            while not game_over:
                # Mudando o jogador
                ply = -ply

                # Selecionando uma jogada aleatória
                sim_move = self.random(local_board)

                # Realiza a jogada
                local_board = self.play_move(local_board, sim_move, ply)

                # Verificas se o jogo acabou
                game_over, score, winner = self.is_terminal(local_board)

            # Salvando os movimentos
            if player == -1 and score <= 0:
                win_moves.append(move)

            if player == 1 and score >= 0:
                win_moves.append(move)

        # Selecionar a jogada com base no melhor resultado
        moves, count = np.unique(win_moves, return_counts=True)
        print(f'Número de vitóridas: {len(win_moves)}\
        \nMovimentos:\n{moves}\nNum Jogadas:\n{count}')

        aval_moves = self.get_availables(board)
        best_move = aval_moves[np.argmax(count)]
        return best_move


#### Função para o jogador humano
def human_play(game):
    print('From function:\nHuman playing...')

    # checking the availables moves
    print(game.get_availables())
    while True:
        move = int(input('Select move:'))
        if move < 1 or move > 9 or not game.is_available(move):
            move = int(input('Invalide move:'))
        else:
            break

        print(f'move: {move}')

    return move


## Função principal do jogo (Main)
def main():
    print('Running ok!')
    board = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
    game = Game()  # board=board)
    player = -1  # human play
    computer = Computer(algorithm='montecarlo')

    Carlos = Computer()
    Moises = Computer(algorithm="MINIMAX")

    game_over = False

    while not game_over:
        game.print_game()

        # For human player
        if player == 1:
            print('Player 1: Carlos')
            game.play_move(human_play(game), player)
            # game.play_move(Carlos.play(board=game.board,player=player,simulations=100),player)

        # For computer player
        if player == -1:
            print('Player 2: Moises')
            game.play_move(Moises.play(board=game.board, player=player, simulations=5000), player)

        game_over = game.is_terminal()

        player = -player

    print(f'Fim de jogo.\nGanhador: {game.winner}\n{game.score}')
    game.print_game()


## Código principal

if __name__ == '__main__':
    main()
