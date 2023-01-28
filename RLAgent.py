import quarto
import numpy as np
import copy
import random

# def quarto_to_tuple(quarto: quarto.Quarto):
#     game = quarto.get_board_status()
#     return tuple([a for x in list(map(list, game)) for a in x])

class RLAgent(quarto.Player):
    """RL Agent"""

    def __init__(self, quarto: quarto.Quarto, learning_rate: float = 0.1, discount: float = 0.95, epsilon: float = 1) -> None:
        super().__init__(quarto)
        self._learning_rate = learning_rate
        self._discount = discount
        self._epsilon = epsilon
        self._BOARDSIZE = 4 * 4

        # Q table indexed by (status, peice) -> array of 16, one value for each move (also not possible ones)
        self._q_table = {}
        self._best_q_table = {}
        self.init_table()

    @property
    def get_qtable(self) -> dict:
        return self._q_table

    def playable_pieces(self) -> list:
        game = self.get_game().get_board_status()
        game = list(map(list, game))
        pieces = list(range(16))
        return [p for p in pieces if p not in [a for l in game for a in l if a != -1]]

    def possible_moves(self) -> list:
        game = self.get_game().get_board_status()
        game = list(map(list, game))
        moves = []
        for y in range(len(game)):
            for x in range(len(game[0])):
                if game[y][x] == -1: moves.append((y, x))
        return moves

    def init_table(self):
        game = quarto.quarto_to_tuple(self.get_game().get_board_status())
        self._q_table[game] = np.random.uniform(low=-2, high=1, size=(16, 16))

    def get_epsilon(self):
        return self._epsilon

    def update_epsilon(self, delta: float) -> None:
        self._epsilon -= delta

    def choose_piece(self) -> int:
        # TODO now selecting the piece that contains a move that has the highest value
        # should try to select the piece that, given that the opponent plays optimally, has
        # the highest reward for me
        q = self.get_game()
        game = quarto.quarto_to_tuple(q.get_board_status())
        max_p = 0
        pieces = q.get_playable_pieces()
        if np.random.random() < self._epsilon:
            max_p = np.random.choice(pieces)
        else:
            max_val = -2
            for p in pieces:
                val = max(self._q_table[game][p])
                if val > max_val:
                    max_val = val
                    max_p = p
        return max_p

    def place_piece(self) -> tuple[int, int]:
        q = self.get_game()
        game = quarto.quarto_to_tuple(q.get_board_status())
        piece = q.get_selected_piece()
        moves = q.get_possible_moves()
        max_move = max_move = random.choice(moves)
        #print(f"possible moves {moves}")
        if np.random.random() > self._epsilon:
            max_val = -2
            for m in moves:
                val = max(self._q_table[game][piece])
                if val > max_val:
                    max_val = val
                    max_move = m
        if max_move == ():
            print("debug")
        return max_move