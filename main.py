import os
import random
import numpy as np
from pathlib import Path
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


def load_model(make_new=True, name=None):
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

    winner = env.winner()
    if render:
        if winner:
            print([agent1, agent2][winner - 1], "is the winner!")
        else:
            print("Tie!")

    return winner, (agent1.learn(), agent2.learn())


def train(episodes=100, render=False):
    model = load_model()

    lr_decay = .9995
    lr_min = .01

    agents = [
        QAgent("AI1", model, epsilon=0.01),
        QAgent("AI2", model, epsilon=0.01),
        QAgent("AI3", TicTacToeTable(lr=.01), epsilon=.9),
        RandomAgent("RNG"),
        # Human("Player1"),
    ]

    games = [
        (agents[0], agents[1]),
        (agents[0], agents[1]),
        (agents[0], agents[1]),
        (agents[0], agents[1]),
        (agents[0], agents[1]),
        (agents[0], agents[1]),
        (agents[0], agents[1]),
        (agents[0], agents[1]),
        (agents[0], agents[1]),
        (agents[0], agents[1]),
        (agents[0], agents[1]),
        (agents[0], agents[1]),
        (agents[0], agents[1]),
        (agents[0], agents[2]),
        (agents[0], agents[2]),
        (agents[0], agents[2]),
        (agents[0], agents[3]),
    ]

    loss_sum = 0
    loss_length = 50
    loss_history = [0] * loss_length
    with tqdm(range(episodes), disable=render) as pbar:
        for _ in pbar:
            # choose two random agents
            agent1, agent2 = random.sample(random.choice(games), 2)
            winner, (agent1.loss, agent2.loss) = play_game(agent1, agent2, render=render)
            winner_name = ['---', agent1.name, agent2.name][winner]

            model_loss = agents[0].loss
            loss_history.append(model_loss)
            loss_sum += model_loss - loss_history.pop(0)

            # display len(model)
            pbar.set_description(
                f"{model.lr=:.3f}, {agents[0].epsilon=:.3f}, {loss_sum=:.3f}, {winner_name=}")

            model.lr *= lr_decay
            model.lr = max(model.lr, lr_min)
            if render and winner and winner_name not in ["AI1", "AI2"]:
                input()

    # visualize_model(new_model)

    model.save(find_model_path())


def play():
    model = load_model(make_new=False, name="human-trained") or load_model()

    model.lr = .5
    agent1 = Human("Player1")
    # agent1 = QAgent("AI-1", model=load_model(), epsilon=0.0)

    agent2 = QAgent("AIH", model=model, epsilon=0.0)
    # agent2 = RandomAgent("X")

    while True:
        agent1, agent2 = random.sample([agent1, agent2], 2)
        winner, (loss1, loss2) = play_game(agent1, agent2)
        model.save(find_model_path(name="human-trained"))


def visualize_model(model, n_columns=15, column_width=10):
    "print all boards in model in order of value"
    boards = [
        (value, state)
        for state, value
        in model.items()
    ]

    boards.sort(key=lambda x: (x[1][1], x[0]), reverse=True)

    num_lines = 4
    for i in range(0, len(boards), n_columns):
        for j, (value, (board, my_turn)) in enumerate(boards[i: i+n_columns]):
            message = visualize_board(board) + "\x1B[9m" * (not my_turn) + f"\n{value:6.2f}\x1B[0m"
            goto_column_start = f"\x1B[{j * column_width}G"

            message = goto_column_start + message.replace("\n", "\n" + goto_column_start)
            message += f"\x1B[{num_lines-1}A"
            # for chr in message:
            #     print(chr, end="")

            print(*message, sep="", end="")

        print("\n" * num_lines)


def main():
    # visualize_model(load_model())
    # train(10000)
    train(1000, render=True)
    # play()


if __name__ == '__main__':
    main()
