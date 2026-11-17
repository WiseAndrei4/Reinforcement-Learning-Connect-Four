import multiprocessing as mp
from FeatureQLearningAgentDriver import FeatureQLearningAgent
from SelfPlayTrainer import SelfPlayTrainer
from QTableAgentDriver import QTableAgent


def worker_train(episodes_per_worker):
    # Train a single, isolated, agent on every core
    local_agent = FeatureQLearningAgent()
    # local_agent = QTableAgent()
    trainer = SelfPlayTrainer(local_agent, random_chance=0.2)
    trainer.run_training(episodes=episodes_per_worker)
    return local_agent.weights
    # return local_agent.q_table


def merge_q_tables(tables):
    """Merge worker Q-tables by averaging non-zero Q-values for shared states."""
    merged = {}
    counts = {}

    for table in tables:
        for state, q_vals in table.items():
            if state not in merged:
                merged[state] = q_vals[:]
                counts[state] = [1 if q != 0.0 else 0 for q in q_vals]
            else:
                for i, q in enumerate(q_vals):
                    if q != 0.0:
                        merged[state][i] += q
                        counts[state][i] += 1

    for state, q_vals in merged.items():
        for i in range(7):
            if counts[state][i] > 1:
                q_vals[i] /= counts[state][i]

    return merged


if __name__ == "__main__":
    num_cores = mp.cpu_count()
    # num_cores = min(8, mp.cpu_count() )
    total_episodes = 20000000  # 20m
    episodes_per_core = total_episodes // num_cores

    print(
        f"Using {num_cores} cores for training,  ({episodes_per_core} episodes/core)..."
    )

    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(worker_train, [episodes_per_core] * num_cores)

    # Learning mean for every core
    """
    master_agent = FeatureQLearningAgent()
    master_agent.weights = [
        sum(w[i] for w in results) / len(results)
        for i in range(len(master_agent.weights))
    ]
    """
    master_agent = FeatureQLearningAgent()
    master_agent.weights = [
        sum(w[i] for w in results) / len(results)
        for i in range(len(master_agent.weights))
    ]
    master_agent.save_weights("weights.json")
    # master_agent = QTableAgent()
    # master_agent.q_table = merge_q_tables(results)
    # master_agent.save_agent("agent.json")

    print("Done! Learned weights:", [round(w, 4) for w in master_agent.weights])
    # print(f"Done! Learned {len(master_agent.q_table)} states.")
