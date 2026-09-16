from functools import reduce
import multiprocessing as mp

# Importăm componentele create anterior
from AgentDriver import QLearningAgent
from SelfPlayTrainer import SelfPlayTrainer
from PlayGame import save_agent


def worker_train(episodes_per_worker):
    """Metodă executată izolat pe un singur nucleu CPU."""
    local_agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)
    trainer = SelfPlayTrainer(local_agent, random_chance=0.2)
    trainer.run_training(episodes=episodes_per_worker)
    return local_agent.q_table


def merge_q_tables(q_table1, q_table2):
    """Îmbină două Q-Tables calculând media scorurilor pentru stările comune."""
    merged = q_table1.copy()
    for state, q_values2 in q_table2.items():
        if state not in merged:
            merged[state] = q_values2
        else:
            # Păstrăm valoarea Q care deviază cel mai mult de la 0.0 (care are experiență reală)
            q_values1 = merged[state]
            best_q = []
            for q1, q2 in zip(q_values1, q_values2):
                best_q.append(q1 if abs(q1) > abs(q2) else q2)
            merged[state] = best_q
    return merged


if __name__ == "__main__":
    num_cores = mp.cpu_count()  # Detectează automat toate nucleele disponibile
    total_episodes = 2000000  #5m
    episodes_per_core = total_episodes // num_cores

    print(
        f"🚀 Pornim antrenarea paralelă pe {num_cores} nuclee ({episodes_per_core} meciuri/nucleu)..."
    )

    # Lansăm procesele în paralel
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(worker_train, [episodes_per_core] * num_cores)

    print("🔄 Îmbinăm rezultatele de pe toate nucleele...")
    final_q_table = reduce(merge_q_tables, results)

    # Salvăm rezultatul final într-o singură instanță de agent
    master_agent = QLearningAgent()
    master_agent.q_table = final_q_table
    save_agent(master_agent, "agent_connect4.pkl")

    print(
        f"✅ Salvat cu succes! Q-Table-ul final conține {len(final_q_table)} stări unice."
    )