import numpy as np

from abc import ABC, abstractmethod


class AgentBase(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def act(self, obs):
        pass

    def reward(self, reward):
        pass

    def reset(self):
        pass

    def learn(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


class RandomAgent(AgentBase):
    def act(self, state: tuple[np.ndarray, bool]) -> int:
        board, my_turn = state
        return np.random.choice(np.where(board == 0)[0])
