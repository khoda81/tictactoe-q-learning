from agents.agent import AgentBase
import numpy as np


class Human(AgentBase):
    def act(self, obs):
        board, turn = obs
        if not turn:
            return

        possible_moves = np.where(board == 0)[0]
        if len(possible_moves) == 1:
            return possible_moves[0]

        action = input(f"{self.name}'s move: ")
        while True:
            try:
                action = int(action) - 1
                assert action in range(9)
                break

            except (ValueError, AssertionError):
                print("Enter a number from 1..9")
                action = input(f"{self.name}'s move: ")

        return [6, 7, 8, 3, 4, 5, 0, 1, 2][action]

    def reward(self, reward):
        if reward > 0:
            print(f"{self.name}: Yay! 🎉")
        elif reward < 0:
            print(f"{self.name}: Boo! 😭")
