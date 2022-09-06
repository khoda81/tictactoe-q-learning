import os
from pathlib import Path
import random
from tqdm import tqdm

from agents import QAgent, Human, RandomAgent
from agents.qagent import TicTacToeTable
from tictactoe_env import TicTacToeEnv, visualize_board


MODEL_PATH = Path(__file__).parent / 'models/'


def find_model_path(path=MODEL_PATH, name=None):
    "find model path"
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = 'model'

    try:
        # find model path
        model_path = next(path.glob(f'*{name}*.qtable'))
    except StopIteration:
        model_path = path / f"{name}.qtable"

    return model_path


def load_model(make_new=False, name=None):
    path = find_model_path(name=name)
    if os.path.exists(path):
        return TicTacToeTable.load(path)
    elif make_new:
        model = TicTacToeTable()
        model.save(path)
        return model


def play_game(agent1, agent2, render=True):
    env = TicTacToeEnv()
    obs1, obs2 = env.reset()

    agent1.reset()
    agent2.reset()

    if render:
        print(agent1, "vs", agent2)
        env.render()

    done = False
    while not done:
        actions = [agent1.act(obs1), agent2.act(obs2)]

        (obs1, obs2), (reward1, reward2), done, info = env.step(actions)

        agent1.reward(reward1)
        agent2.reward(reward2)

        if render:
            print("-" * 5)
            env.render()

    if render:
        winner = env.winner()
        if winner:
            print([agent1, agent2][winner - 1], "is the winner!")
        else:
            print("Tie!")

    agent1.learn()
    agent2.learn()

    return winner


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

    model.save(find_model_path())


def play():
    model = load_model(name="human-trained") or load_model()

    model.lr = .5
    agent1 = Human("Player1")
    agent2 = QAgent("AI", model=model, epsilon=0.0)
    # agent2 = RandomAgent("X")

    while True:
        agent1, agent2 = random.sample([agent1, agent2], 2)
        play_game(agent1, agent2)
        model.save(find_model_path(name="human-trained"))


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
