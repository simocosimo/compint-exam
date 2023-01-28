class Move(object):
    def __init__(self, position: tuple, piece: int, next_piece: int):
        self._position = position
        self._piece = piece
        self._next_piece = next_piece
    
    def __str__(self):
        return f"Pos: {self._position} - Current piece: {self._piece} - Next piece: {self._next_piece}"

    def __repr__(self):
        return f"Pos: {self._position} - Current piece: {self._piece} - Next piece: {self._next_piece}"

    def get_position(self):
        return self._position

    def get_next_piece(self):
        return self._next_piece

    def get_current_piece(self):
        return self._piece