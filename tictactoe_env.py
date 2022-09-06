import numpy as np
import gym
from gym.spaces import Discrete, Tuple, Box


ICONS = ['⬜', '❌', '⭕']


def visualize_board(board):
    return '\n'.join(
        ''.join(
            ICONS[cell]
            for cell in board[i:i+3]
        )
        for i in [0, 3, 6]
    )


class TicTacToeEnv(gym.Env):
    def __init__(self):
        self.action_space = Discrete(9)
        self.observation_space = Tuple([
            Tuple([
                Box(0, 2, (9,)),
                Discrete(2),
            ]),
            Tuple([
                Box(0, 2, (9,)),
                Discrete(2),
            ]),
        ])

    def reset(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.turn = False

        return self.observation()

    def step(self, actions):
        self.update(actions[self.turn])
        winner = self.winner()
        rewards = [0, 0]
        if winner == 1:
            rewards = [1, -1]
        elif winner == 2:
            rewards = [-1, 1]

        return self.observation(), rewards, winner is not None, {}

    def render(self):
        print(visualize_board(self.board))

    def update(self, place):
        if place not in range(9) or self.board[place]:
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
        """
        Returns a tuple of two observations, one for each player.
        Observations are normalized such that each player thinks it is X.
        """
        obs1 = self.board.copy()

        obs2 = self.board.copy()
        mask = np.where(obs2 != 0)
        obs2[mask] = ((obs2[mask] - 1) ^ 1) + 1

        return (
            (obs1, not self.turn),
            (obs2, self.turn),
        )
