import os
from pathlib import Path
import random
from tqdm import tqdm

from agents import QAgent, Human, RandomAgent
from agents.qagent import TicTacToeTable
from tictactoe_env import ICONS, TicTacToeEnv, visualize_board


def load_model():
    "load model if exists"
    if os.path.exists(MODEL_PATH):
        model = TicTacToeTable.load(MODEL_PATH)
    else:
        model = TicTacToeTable()
        model.sa1ve(MODEL_PATH)

    return model


def play_game(agent1, agent2, render=True):
    env = TicTacToeEnv()

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


MODEL_PATH = Path(__file__).parent / 'models/'


def find_model_path(path=MODEL_PATH, name=None):
    "find model path"
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # find model path
    if name is None:
        model_path = next(path.glob('*.qtable'))
    else:
        model_path = next(path.glob(f'{name}*.qtable'))

    return model_path


def train(episodes=100, render=False):
    model = load_model()

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
            winner = ['---', agent1.name, agent2.name][play_game(agent1, agent2, render=render)]

            # display len(model)
            pbar.set_description(f"{model.lr=:.3f}, {agents[0].epsilon=:.3f}, {winner=}")

            model.lr *= lr_decay
            model.lr = max(model.lr, lr_min)

    visualize_model(new_model)

    model.save(MODEL_PATH)


def play():
    model = load_model()

    model.lr = .5
    agent1 = Human("Player1")
    agent2 = QAgent("AI", model=model, epsilon=0.0)
    # agent2 = RandomAgent("X")

    while True:
        agent1, agent2 = random.sample([agent1, agent2], 2)
        play_game(agent1, agent2)
        model.save(MODEL_PATH)


def visualize_model(model):
    "print all boards in model in order of value"
    boards = [
        (value, board)
        for board, value in model.items()
    ]

    boards.sort(key=lambda x: x[0], reverse=True)

    for value, board in boards:
        print(visualize_board(board))
        print(f"{value:6.2f}")
        print()


def main():
    # train(10000)
    # train(10, render=True)
    play()


if __name__ == '__main__':
    main()
