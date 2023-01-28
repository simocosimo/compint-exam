# Free for personal or classroom use; see 'LICENSE.md' for details.
# https://github.com/squillero/computational-intelligence

import numpy as np
from abc import abstractmethod
import copy
from move import Move
import random
import math

class Player(object):

    def __init__(self, quarto) -> None:
        self.__quarto = quarto

    @abstractmethod
    def choose_piece(self) -> int:
        pass

    @abstractmethod
    def place_piece(self) -> tuple[int, int]:
        pass

    def get_game(self):
        return self.__quarto


class Piece(object):

    def __init__(self, high: bool, coloured: bool, solid: bool, square: bool) -> None:
        self.HIGH = high
        self.COLOURED = coloured
        self.SOLID = solid
        self.SQUARE = square
        self.binary = [int(high), int(coloured), int(solid), int(square)]


class Quarto(object):

    MAX_PLAYERS = 2
    BOARD_SIDE = 4

    def __init__(self) -> None:
        self.__players = ()
        self.reset()

    def reset(self):
        self._board = np.ones(
            shape=(self.BOARD_SIDE, self.BOARD_SIDE), dtype=int) * -1
        self._binary_board = np.full(
            shape=(self.BOARD_SIDE, self.BOARD_SIDE, 4), fill_value=np.nan)
        self.__pieces = []
        self.__pieces.append(Piece(False, False, False, False))  # 0
        self.__pieces.append(Piece(False, False, False, True))  # 1
        self.__pieces.append(Piece(False, False, True, False))  # 2
        self.__pieces.append(Piece(False, False, True, True))  # 3
        self.__pieces.append(Piece(False, True, False, False))  # 4
        self.__pieces.append(Piece(False, True, False, True))  # 5
        self.__pieces.append(Piece(False, True, True, False))  # 6
        self.__pieces.append(Piece(False, True, True, True))  # 7
        self.__pieces.append(Piece(True, False, False, False))  # 8
        self.__pieces.append(Piece(True, False, False, True))  # 9
        self.__pieces.append(Piece(True, False, True, False))  # 10
        self.__pieces.append(Piece(True, False, True, True))  # 11
        self.__pieces.append(Piece(True, True, False, False))  # 12
        self.__pieces.append(Piece(True, True, False, True))  # 13
        self.__pieces.append(Piece(True, True, True, False))  # 14
        self.__pieces.append(Piece(True, True, True, True))  # 15
        self._current_player = 0
        self.__selected_piece_index = -1

    def set_players(self, players: tuple[Player, Player]):
        self.__players = players

    def get_current_player(self) -> int:
        '''
        Gets the current player
        '''
        return self._current_player

    def select(self, pieceIndex: int) -> bool:
        '''
        select a piece. Returns True on success
        '''
        if pieceIndex not in self._board:
            self.__selected_piece_index = pieceIndex
            return True
        return False

    def place(self, x: int, y: int) -> bool:
        '''
        Place piece in coordinates (x, y). Returns true on success
        '''
        if self.__placeable(x, y):
            self._board[y, x] = self.__selected_piece_index
            self._binary_board[y,
                               x][:] = self.__pieces[self.__selected_piece_index].binary
            return True
        return False

    def __placeable(self, x: int, y: int) -> bool:
        return not (y < 0 or x < 0 or x > 3 or y > 3 or self._board[y, x] >= 0)

    def print(self):
        '''
        Print the board
        '''
        for row in self._board:
            print("\n -------------------")
            print("|", end="")
            for element in row:
                print(f" {element: >2}", end=" |")
        print("\n -------------------\n")
        print(f"Selected piece: {self.__selected_piece_index}\n")

    def get_piece_charachteristics(self, index: int) -> Piece:
        '''
        Gets charachteristics of a piece (index-based)
        '''
        return copy.deepcopy(self.__pieces[index])

    def get_board_status(self) -> np.ndarray:
        '''
        Get the current board status (pieces are represented by index)
        '''
        return copy.deepcopy(self._board)

    def get_selected_piece(self) -> int:
        '''
        Get index of selected piece
        '''
        return copy.deepcopy(self.__selected_piece_index)

    def __check_horizontal(self) -> int:
        hsum = np.sum(self._binary_board, axis=1)

        if self.BOARD_SIDE in hsum or 0 in hsum:
            return self._current_player
        else:
            return -1

    def __check_vertical(self):
        vsum = np.sum(self._binary_board, axis=0)

        if self.BOARD_SIDE in vsum or 0 in vsum:
            return self._current_player
        else:
            return -1

    def __check_diagonal(self):
        dsum1 = np.trace(self._binary_board, axis1=0, axis2=1)
        dsum2 = np.trace(np.fliplr(self._binary_board), axis1=0, axis2=1)

        if self.BOARD_SIDE in dsum1 or self.BOARD_SIDE in dsum2 or 0 in dsum1 or 0 in dsum2:
            return self._current_player
        else:
            return -1

    def check_winner(self) -> int:
        '''
        Check who is the winner
        '''
        l = [self.__check_horizontal(), self.__check_vertical(),
             self.__check_diagonal()]
        for elem in l:
            if elem >= 0:
                return elem
        return -1

    def check_finished(self) -> bool:
        '''
        Check who is the loser
        '''
        for row in self._board:
            for elem in row:
                if elem == -1:
                    return False
        return True

    def run(self) -> int:
        '''
        Run the game (with output for every move)
        '''
        winner = -1
        while winner < 0 and not self.check_finished():
            self.print()
            piece_ok = False
            while not piece_ok:
                piece_ok = self.select(
                    self.__players[self._current_player].choose_piece())
            piece_ok = False
            self._current_player = (
                self._current_player + 1) % self.MAX_PLAYERS
            self.print()
            while not piece_ok:
                x, y = self.__players[self._current_player].place_piece()
                piece_ok = self.place(x, y)
            winner = self.check_winner()
        self.print()
        return winner


class CustomQuarto(Quarto):

    def __init__(self) -> None:
        super().__init__()

    def set_current_player(self, player_id: int) -> None:
        self._current_player = player_id

    def get_playable_pieces(self):
        game = self.get_board_status()
        pieces = list(range(16))
        return [p for p in pieces if p not in [a for l in game for a in l if a != -1] and p != self.get_selected_piece()]

    def get_possible_moves(self):
        game = self.get_board_status()
        moves = []
        positions = []
        for y in range(len(game)):
            for x in range(len(game[0])):
                if game[y][x] == -1: positions.append((x, y))

        # If there is a move but pieces choise is empty
        # It means there is a forced move and then game ends
        if len(positions) == 1:
            moves.append(Move(
                positions[0], 
                self.get_selected_piece(), 
                0   # doesn't matter
            ))
            return moves

        for p in positions:
            for piece in self.get_playable_pieces():
                moves.append(Move(p, self.get_selected_piece(), piece))
        return moves

    def apply_move(self, move: Move):
        pos = move.get_position()
        if not self.place(pos[0], pos[1]): 
            print("Invalid move")
            exit(1)
        self._Quarto__selected_piece_index = move._next_piece
        self._Quarto_current_player = 1 - self.get_current_player()

    def winning_move(self):
        for current_move in self.get_possible_moves():
            tmp = copy.deepcopy(self)
            tmp.apply_move(current_move)
            iswin = tmp.check_winner()
            if iswin >= 0:
                return current_move
        return None

    def random_run(self) -> int:
        '''
        Run the game (with output for every move)
        '''
        winner = -1
        while winner < 0 and not self.check_finished():
            piece_ok = False
            while not piece_ok:
                piece_ok = self.select(random.randint(0, 15))
            piece_ok = False
            self._Quarto_current_player = (self.get_current_player() + 1) % self.MAX_PLAYERS
            while not piece_ok:
                x, y = random.randint(0, 3), random.randint(0, 3)
                piece_ok = self.place(x, y)
            winner = self.check_winner()
        return winner

class MCTSNode(object):
    
    def __init__(self, 
        game: Quarto, 
        move: Move = None, 
        parent: object = None,
        rollouts: int = 0
    ):
        self._game = game
        self._parent = parent
        self._node_move = move
        self._children = []
        self._num_rollouts = rollouts
        self._unvisited_moves = game.get_possible_moves()
        self._win_counts = {}
        self._win_counts["AGENT"] = 0
        self._win_counts["OPPONENT"] = 0

    def random_legal_move(self) -> Move:
        index = random.randrange(len(self._unvisited_moves))
        ret = copy.deepcopy(self._unvisited_moves[index])
        del self._unvisited_moves[index]
        return ret

    def can_add_child(self) -> bool:
        return len(self._unvisited_moves) > 0

    def is_terminal(self) -> bool:
        return self._game.check_finished() or self._game.check_winner() >= 0

    def winning_fraction(self, player: str) -> float:
        if player == 1:
            # I won
            player = "AGENT"
        elif player == 0:
            # opponent won
            player = "OPPONENT"
        else:
            # random value just to crash
            player = "LUL"
        return self._win_counts[player] / self._num_rollouts

    def record_win(self, winner: str):
        count = 0
        if winner in self._win_counts:
            count = self._win_counts[winner]
            self._win_counts[winner] = count + 1
        self._num_rollouts += 1
    
    def propagate_wins(self, winner: str):
        self.record_win(winner)

        if self._parent is not None:
            parent = self._parent
            parent.propagate_wins(winner)
    

class MCTSAgent(Player):
    """MonteCarlo Agent"""

    def __init__(self, quarto: Quarto, num_rounds: int = 1000, c: float = 1.0):
        super().__init__(quarto)
        # This is a number used to limit the MC tree search
        self._num_rounds = num_rounds
        self._c = c
        self._picked_move = None

    def choose_piece(self) -> int:
        # If playing as player 0 I decide the first piece and it will be always piece 0 
        # (no metter on the choice when choosing first piece)
        # If playing as player 1 the field will not be None because after moving I select the next piece
        ret = copy.deepcopy(self._picked_move._next_piece) if self._picked_move is not None else 0
        return ret

    def place_piece(self) -> tuple[int, int]:
        self.select_move(self.get_game())
        return self._picked_move._position

    def select_move(self, state: Quarto):
        root = MCTSNode(state)

        winning_move = root._game.winning_move()
        if winning_move is not None: 
            self._picked_move = winning_move
            return

        for _ in range(self._num_rounds):
            self.execute_round(root)

        # Performed all rounds, time to choose a move
        self._picked_move = self.pick_best_move(root)
    
    def execute_round(self, root: MCTSNode):
        node = root

        # Find a node to add a child
        # If cannot add child
        # AND
        # state is not terminal (board is full or winner)
        while not node.can_add_child() and not node.is_terminal():
            node = self.select_child(node)
        
        # Add a new move into the tree
        if node.can_add_child():
            node = self.add_child_for_random_move(node)
        
        # Simulate a random game for this node
        winner = self.simulate_random_game(node._game)
        node.propagate_wins(winner)

    def select_child(self, node: MCTSNode) -> MCTSNode:
        total_rollouts = 0
        for child in node._children:
            total_rollouts += child._num_rollouts
        
        best_score = -1.0
        best_child = None
        for child in node._children:
            uct_score = self.calculate_uct_score(
                total_rollouts,
                child._num_rollouts,
                child.winning_fraction(node._game.get_current_player())
            )

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        if best_child is None:
            print("Child not found")
            exit(1)
        return best_child

    # Calculate upper confidence bound fo trees (UCT)
    # This gives you a balance between exploration (breadth) and exploitation (depth)
    def calculate_uct_score(self, parent_roullouts: float, child_rollouts: float, win_pct: float) -> float:
        exploration = math.sqrt(math.log10(parent_roullouts) / child_rollouts)
        return win_pct + self._c * exploration

    def add_child_for_random_move(self, node: MCTSNode) -> MCTSNode:
        tmp = copy.deepcopy(node._game)
        next_move = node.random_legal_move()
        tmp.apply_move(next_move)
        child = MCTSNode(tmp, parent=node, move=next_move)
        node._children.append(child)
        return child

    def simulate_random_game(self, game: Quarto):
        rgame = copy.deepcopy(game)
        return rgame.random_run()

    def pick_best_move(self, node: MCTSNode) -> Move:
        best_move = None
        best_percent = -1.0

        for child in node._children:
            if self.is_losing_move(child, node):
                continue

            child_percent = child.winning_fraction(node._game.get_current_player())

            if child_percent > best_percent:
                #print(f"New best move found with {child_percent} value")
                best_percent = child_percent
                best_move = child._node_move
        
        if best_move is None:
            best_move = node._children[0]._node_move
        
        return best_move

    def is_losing_move(self, child: MCTSNode, parent: MCTSNode) -> bool:
        child_move = child._node_move
        tmp = copy.deepcopy(parent._game)
        tmp.apply_move(child_move)
        for move in tmp.get_possible_moves():
            lgame = copy.deepcopy(tmp)
            lgame.apply_move(move)
            winner = lgame.check_winner()
            if winner == 0 or lgame.check_finished():
                return True
        return False
