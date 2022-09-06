
from functools import cache
from pathlib import Path
from typing import Union
import numpy as np
from collections import defaultdict
import pickle
from collections import namedtuple

from agents.agent import AgentBase


class TicTacToeTable:
    mask = np.array([
        [1,    3,    9],
        [27,   81,  243],
        [729, 2187, 6561],
    ])

    def __init__(self, lr=.1):
        self.qtable = defaultdict(float)
        self.lr = lr

    @staticmethod
    def encode_board(new_board):
        return TicTacToeTable._encode_board(tuple(new_board))

    @staticmethod
    @cache
    def _encode_board(new_board):
        new_board = np.array(new_board).reshape(3, 3)
        # find minimum id for board
        ids = []

        # generate all symmetries
        for i in range(4):
            new_board = np.rot90(new_board)

            # e, r, r2, r3
            ids.append(np.sum(TicTacToeTable.mask * new_board))

            # ty, tAC, tx, tBD
            ids.append(np.sum(TicTacToeTable.mask * np.fliplr(new_board)))

        return min(ids)

    @staticmethod
    def decode_board(id):
        board = []
        for i in range(9):
            id, r = divmod(id, 3)
            board.append(r)

        return np.array(board)

    @staticmethod
    def to_id(state: tuple[np.ndarray, bool], action: int) -> int:
        "Generate unique id for board and action pair"
        board, my_turn = state
        if my_turn:
            new_board = board.copy()
            assert not new_board[action]

            new_board[action] = 1
            return TicTacToeTable.encode_board(new_board)
        else:
            return TicTacToeTable.encode_board(board)

    def update(self, state, action, target):
        delta = target - self[state, action]
        self[
            state,
            action
        ] += self.lr * delta

    def items(self):
        for board, value in self.qtable.items():
            yield self.decode_board(board), value

    @staticmethod
    def load(path: Union[str, Path]) -> 'TicTacToeTable':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path: Union[str, Path]):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def __call__(self, board: np.ndarray, action: int) -> float:
        return self[board, action]

    def __getitem__(self, key: tuple[np.ndarray, int]):
        state, action = key
        return self.qtable.__getitem__(self.to_id(state, action))

    def __setitem__(self, key: tuple[np.ndarray, int], value):
        state, action = key
        return self.qtable.__setitem__(self.to_id(state, action), value)

    def __delitem__(self, key: tuple[np.ndarray, int]):
        state, action = key
        return self.qtable.__delitem__(self.to_id(state, action))

    def __contains__(self, key: tuple[np.ndarray, int]):
        state, action = key
        return self.qtable.__contains__(self.to_id(state, action))

    def __len__(self):
        return self.qtable.__len__()

    def __iter__(self):
        for board in self.qtable:
            yield self.decode_board(board)

    def __repr__(self):
        return f'TicTacToeTable(lr={self.lr})'


Gameplay = namedtuple('Gameplay', ['states', 'actions', 'rewards'])


class QAgent(AgentBase):
    gameplays: list[Gameplay]
    current_gameplay: Gameplay

    def __init__(
            self, name, model=None, epsilon=0.2, gamma=0.99, max_gameplays=2,
            decay_rate=1e-2):
        super().__init__(name)

        self.gameplays = []

        if model is None:
            model = TicTacToeTable(lr=0.5)

        self.model = model
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_rate = decay_rate

        self.max_gameplays = max_gameplays

    def _act(self, state: tuple[np.ndarray, bool]) -> int:
        best_move = (-float('inf'), None)
        for move in self.possible_moves(state):
            new_move = self.model(state, move), move
            best_move = max(best_move, new_move)

        return best_move[1]

    def possible_moves(self, state):
        board, my_turn = state
        if not my_turn:
            return [None]

        return np.where(board == 0)[0]

    def reset(self) -> None:
        """Set up the agent for a new game"""
        self.current_gameplay = Gameplay([], [], [])
        self.gameplays.append(self.current_gameplay)

        if len(self.gameplays) > self.max_gameplays:
            self.gameplays.pop(0)

    def act(self, state: tuple[np.ndarray, bool]) -> int:
        """Find an epsilon-greedy action and remember state and action"""
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.possible_moves(state))
            self.epsilon *= 1 - self.decay_rate
        else:
            action = self._act(state)

        try:
            self.current_gameplay.states.append(state)
            self.current_gameplay.actions.append(action)
            self.current_gameplay.rewards.append(0)
        except AttributeError as e:
            # add help message to the exception before raising it
            e.args = (e.args[0] + ", Perhaps you didn't call agent.reset() before calling act()?",)
            raise e

        return action

    def reward(self, reward: float) -> None:
        """Give a reward for last action performed by the agent"""
        self.current_gameplay.rewards[-1] += reward

    def learn(self) -> None:
        """Train on all gameplays and update q-model"""

        for gameplay in self.gameplays:
            gameplay = reversed(list(zip(gameplay.states, gameplay.actions, gameplay.rewards)))
            state, _, reward = next(gameplay)
            for prev_state, prev_action, prev_reward in gameplay:
                max_q = max(self.model(state, i) for i in self.possible_moves(state))
                q_target = reward + self.gamma * max_q
                self.model.update(prev_state, prev_action, q_target)
                state, _, reward = prev_state, prev_action, prev_reward

    def load_pretrained(self, path: Union[str, Path] = None) -> None:
        """Load the model from a file"""
        path = path or f"{self.name}.qtable"
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def save_pretrained(self, path: Union[str, Path] = None) -> None:
        """Save the model to a file"""
        path = path or f"{self.name}.qtable"
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
