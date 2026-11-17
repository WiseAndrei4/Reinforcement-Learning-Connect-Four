import random
import time
from GameDriver import GameDriver
from FeatureQLearningAgentDriver import FeatureQLearningAgent
from QTableAgentDriver import QTableAgent
from MinMaxAgentDriver import MinMaxAgent


class RandomAgent:
    def choose_action(self, board, my_symbol="A"):
        valid = [c for c in range(7) if board[0][c] == 0]
        return random.choice(valid) if valid else 0


def play_match(agent_a, agent_b):
    env = GameDriver()
    times_a, times_b = [], []

    while env.winner == -1:
        state_copy = [row[:] for row in env.board]
        if env.step == 1:
            t0 = time.perf_counter()
            col = agent_a.choose_action(state_copy, my_symbol="A")
            times_a.append((time.perf_counter() - t0) * 1000.0)
        else:
            t0 = time.perf_counter()
            col = agent_b.choose_action(state_copy, my_symbol="B")
            times_b.append((time.perf_counter() - t0) * 1000.0)

        res = env.play(col)
        if res == "gresit":
            # Penalize illegal move as immediate loss
            env.winner = 2 if env.step == 1 else 1
            break

    return env.winner, times_a, times_b


def evaluate_pair(name_1, agent_1, name_2, agent_2, games=200):
    wins_1 = wins_2 = draws = 0
    latencies_1, latencies_2 = [], []
    half = games // 2

    # Half games as Player 1 (A), half as Player 2 (B)
    for i in range(games):
        if i < half:
            winner, t1, t2 = play_match(agent_1, agent_2)
            if winner == 1:
                wins_1 += 1
            elif winner == 2:
                wins_2 += 1
            else:
                draws += 1
            latencies_1.extend(t1)
            latencies_2.extend(t2)
        else:
            winner, t2, t1 = play_match(agent_2, agent_1)
            if winner == 2:
                wins_1 += 1
            elif winner == 1:
                wins_2 += 1
            else:
                draws += 1
            latencies_1.extend(t1)
            latencies_2.extend(t2)

    avg_ms_1 = sum(latencies_1) / len(latencies_1) if latencies_1 else 0.0
    avg_ms_2 = sum(latencies_2) / len(latencies_2) if latencies_2 else 0.0

    print(
        f"| {name_1} vs {name_2} | {games} | "
        f"{wins_1 / games * 100:.1f}% | {draws / games * 100:.1f}% | {wins_2 / games * 100:.1f}% | "
        f"{avg_ms_1:.3f} ms vs {avg_ms_2:.3f} ms |"
    )


if __name__ == "__main__":
    rand_agent = RandomAgent()

    qtable_agent = QTableAgent(epsilon=0.0)
    qtable_agent.load_agent("agent.json")
    qtable_agent.epsilon = 0.0  # No exploring

    feat_agent = FeatureQLearningAgent(epsilon=0.0)
    feat_agent.load_weights("weights.json")
    feat_agent.epsilon = 0.0  # No exploring

    minmax_d2 = MinMaxAgent(depth=2)
    minmax_d4 = MinMaxAgent(depth=4)

    print("\n| Matchup (P1 vs P2) | Games | P1 Win | Draw | P2 Win | Avg Inference / Move |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    evaluate_pair("Q-Table", qtable_agent, "Random", rand_agent, games=500)
    evaluate_pair("Feature Q-Learning", feat_agent, "Random", rand_agent, games=500)
    evaluate_pair("Feature Q-Learning", feat_agent, "Q-Table", qtable_agent, games=500)
    evaluate_pair("Feature Q-Learning", feat_agent, "MinMax (d=2)", minmax_d2, games=200)
    evaluate_pair("Feature Q-Learning", feat_agent, "MinMax (d=4)", minmax_d4, games=100)
    evaluate_pair("MinMax (d=4)", minmax_d4, "Random", rand_agent, games=100)
