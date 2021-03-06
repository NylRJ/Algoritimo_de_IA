import numpy as np
import random, copy


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

    def utility(board):
        xx = winner(board)
        if (xx == X):
            return 1
        elif (xx == O):
            return -1
        else:
            return 0

    def winner(board):
        rows = board + get_diagonal(board) + get_columns(board)
        for row in rows:
            current_palyer = row[0]
            if current_palyer is not None and three_in_a_row(row):
                return current_palyer
        return None

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
        if any(np.sum(board, axis=0) == 3) or \
                any(np.sum(board, axis=1) == 3) or \
                sum(np.diag(board)) == 3 or \
                sum(np.diag(board[::-1])) == 3:

            game_over = True
            winner = 1
            score = (len(self.get_availables(board)) + 1)

        # verificando se o jogador 2 venceu
        elif any(np.sum(board, axis=0) == -3) or \
                any(np.sum(board, axis=1) == -3) or \
                sum(np.diag(board)) == -3 or \
                sum(np.diag(board[::-1])) == -3:

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
        # Verificando se ?? um estado terminal
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
        # Verificando se ?? um estado terminal
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

    def alpha_beta(self, board, minimizing=True):
        if self.is_terminal(board)[0]:
            return None

        if minimizing:
            print('Minimizando')
            ply = -1
            return max_alpha_beta_pruning(board, float("-inf"), float("inf"))[1]

        else:
            print('Maximizando')
            ply = 1
            return min_alpha_beta_pruning(board, float("-inf"), float("inf"))[1]

    def max_alpha_beta_pruning(board, alpha, beta):

        if (self.is_terminal(board)[0] == True):
            return utility(board), None
        vall = float("-inf")
        best = None
        for action in actions(board):
            min_val = min_alpha_beta_pruning(result(board, action), alpha, beta)[0]
            if (min_val > vall):
                best = action
                vall = min_val
            alpha = max(alpha, vall)
            if (beta <= alpha):
                break
        return vall, best

    def min_alpha_beta_pruning(board, alpha, beta):
        if (self.is_terminal(board)[0] == True):
            return utility(board), None
        vall = float("inf")
        best = None
        for action in actions(board):
            max_val = max_alpha_beta_pruning(result(board, action), alpha, beta)[0]
            if (max_val < vall):
                best = action
                vall = max_val
            beta = min(beta, vall)
            if (beta <= alpha):
                break
        return vall, best

    def montecarlo(self, board, player=-1, simulations=None):
        # Definindo a ??ltima jogada
        if len(self.get_availables(board)) == 1:
            return self.random(board)

        # Definindo a quantidade de simula????es
        if simulations is None:
            simulations = 1000

        # Vetor para salvar as jogadas
        win_moves = []

        # Realizando simula????o de jogadas
        for i in range(simulations):

            # Inicializando a simula????o
            ply = player
            local_board = copy.copy(board)

            # Selecionar o primeiro movimento utilizando a pol??tica aleat??ria
            move = self.random(local_board)

            # Expans??o do estado
            local_board = self.play_move(local_board, move, ply)
            game_over, score, winner = self.is_terminal(local_board)

            # Realizando a simula????o do jogo
            while not game_over:
                # Mudando o jogador
                ply = -ply

                # Selecionando uma jogada aleat??ria
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
        print(f'N??mero de vit??ridas: {len(win_moves)}\
            \nMovimentos:\n{moves}\nNum Jogadas:\n{count}')

        aval_moves = self.get_availables(board)
        best_move = aval_moves[np.argmax(count)]
        return best_move
