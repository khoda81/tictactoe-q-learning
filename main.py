import os
from pathlib import Path
import pickle
import random
import numpy as np
from tqdm import tqdm

from agent import Agent
from qagent import QAgent, TicTacToeTable, RandomAgent

ICONS = ['.', 'X', 'O']


class Human(Agent):
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
        else:
            print(f"{self.name}: Meh! 😐")


class TicTacToe:
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
        return '\n'.join(
            ' '.join(
                ICONS[cell]
                for cell in self.board[i:i+3]
            )
            for i in [0, 3, 6]
        )


def gameplay(agent1, agent2, render=True):
    env = TicTacToe()

    agent1.reset()
    agent2.reset()

    if render:
        print(agent1, "vs", agent2)
        env.render()

    while (winner := env.winner()) is None:
        obs1, obs2 = env.observation()
        actions = [agent1.act(obs1), agent2.act(obs2)]

        env.update(actions[env.turn])
        if render:
            print("-" * 5)
            env.render()

    if winner:
        [agent1, agent2][winner - 1].reward(1)
        [agent2, agent1][winner - 1].reward(-1)

        if render:
            print([agent1, agent2][winner - 1], "is the winner!")
    else:
        if render:
            print("Tie!")

    agent1.learn()
    agent2.learn()

    return winner


MODEL_PATH = Path(__file__).parent / 'models/model.qtable'


def train(episodes=100, render=False):
    # load model if exists
    if os.path.exists(MODEL_PATH):
        model = TicTacToeTable.load(MODEL_PATH)
    else:
        model = TicTacToeTable()
        model.save(MODEL_PATH)

    model.lr = .3
    lr_decay = .9999
    lr_min = .01

    new_model = TicTacToeTable()

    agents = [
        QAgent("AI1", model),
        # QAgent("AI2", model),
        QAgent("AI3", new_model),
        RandomAgent("RNG"),
        # Human("Player1"),
    ]

    # choose two random agents
    with tqdm(range(episodes), disable=render) as pbar:
        for _ in pbar:
            agent1, agent2 = random.sample(agents, 2)
            winner = ['---', agent1.name, agent2.name][gameplay(agent1, agent2, render=render)]

            # display len(model)
            pbar.set_description(f"{model.lr=:.3f}, {agents[0].epsilon=:.3f}, {winner=}")

            model.lr *= lr_decay
            model.lr = max(model.lr, lr_min)

    visualize_model(new_model)

    model.save(MODEL_PATH)


def play():
    model = TicTacToeTable.load(MODEL_PATH)

    agent1 = Human("Player1")
    agent2 = QAgent("AI", model=model, epsilon=0.0)
    # agent2 = RandomAgent("X")

    while True:
        agent1, agent2 = random.sample([agent1, agent2], 2)
        gameplay(agent1, agent2)
        model.save(MODEL_PATH)


def visualize_model(model):
    "print all boards in model in order of value"
    boards = [
        (value, board)
        for board, value in model.items()
    ]

    boards.sort(key=lambda x: x[0], reverse=True)

    for value, board in boards:
        board = [ICONS[i] for i in board]
        print(*board[0:3])
        print(*board[3:6])
        print(*board[6:9])
        print(f"{value:6.2f}")
        print()


def main():
    # train(10000)
    # train(10, render=True)
    play()


if __name__ == '__main__':
    main()
