

from abc import ABC, abstractmethod


class Agent(ABC):
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
