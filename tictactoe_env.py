import numpy as np


ICONS = ['⬜', '❌', '⭕']


def visualize_board(board):
    return '\n'.join(
        ''.join(
            ICONS[cell]
            for cell in board[i:i+3]
        )
        for i in [0, 3, 6]
    )


class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.turn = False

    def update(self, place):
        if place not in range(9):
            return

        if self.board[place]:
            return

        self.board[place] = self.turn + 1
        self.turn = not self.turn

    def winner(self):
        board = self.board.reshape(3, 3)

        for player in [1, 2]:
            mask = board == player
            out = mask.all(0).any() | mask.all(1).any()
            out |= np.diag(mask).all() | np.diag(mask[:, ::-1]).all()
            if out:
                return player

        if board.all():
            return 0

    def observation(self):
        obs2 = self.board.copy()
        mask = np.where(obs2 != 0)
        obs2[mask] = ((obs2[mask] - 1) ^ 1) + 1

        return (
            (self.board.copy(), not self.turn),
            (obs2, self.turn),
        )

    def render(self):
        print(self)

    def __repr__(self):
        return visualize_board(self.board)
