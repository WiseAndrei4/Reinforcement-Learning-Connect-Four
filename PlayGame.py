import json
from GameDriver import GameDriver
from AgentDriver import QLearningAgent

def save_agent(agent, filename="agent.json"):
    with open(filename, "w") as f:
        json.dump(agent.q_table, f)
    print(f"Agentul a fost salvat în '{filename}'!")
def load_agent(agent, filename="agent.json"):
    try:
        with open(filename, "r") as f:
            agent.q_table=json.load(f)
            print("Succesfully loaded!")
    except FileExistsError as e:
        print("Missing file!")

def show_board(board):
    print("\n   0   1   2   3   4   5   6")  # Indicii coloanelor
    print(" ┌───┬───┬───┬───┬───┬───┬───┐")
    for row in board:
        lila = []
        for cell in row:
            if cell == 0:
                lila.append(" ")  # Spațiu gol
            elif cell == 1:
                lila.append("🔴")  # Jucătorul 1 (Omul)
            else:
                lila.append("🔵")  # Jucătorul 2 (Agentul)
        print(" │ " + " │ ".join(lila) + " │")
        if row != board[-1]:
            print(" ├───┼───┼───┼───┼───┼───┼───┤")
    print(" └───┴───┴───┴───┴───┴───┴───┘")

def play_game(agent):
        # Dezactivăm explorarea pentru meciul oficial (vrem să joace la capacitate maximă)
        agent.epsilon = 0.0

        env = GameDriver()
        print("\n🎮 MECI ÎNCEPUT: Tu ești cu ROȘU (🔴, Primul), Agentul este cu ALBASTRU (🔵, Al doilea).")

        while env.winner == -1:
            show_board(env.board)

            if env.step == 1:
                # Rândul tău
                try:
                    col = int(
                        input("\n👉 Alege o coloană unde să pui piesa (0-6): ")
                    )
                    if col < 0 or col > 6:
                        print("❌ Coloană invalidă! Introdu un număr între 0 și 6.")
                        continue
                except ValueError:
                    print("❌ Te rog introdu un număr valid.")
                    continue

                rezultat = env.play(col)
                if rezultat == "gresit":
                    print("❌ Coloana este plină! Alege altă coloană.")
                    continue
            else:
                # Rândul Agentului AI
                print("\n🤖 Agentul se gândește...")
                state_copy = [row[:] for row in env.board]
                col_ai = agent.choose_action(state_copy)

                rezultat = env.play(col_ai)
                # Dacă agentul încearcă o mutare greșită (rar, dar posibil dacă nu e complet antrenat)
                # îl forțăm să aleagă o coloană liberă
                while rezultat == "gresit":
                    col_ai = (col_ai + 1) % 7
                    rezultat = env.play(col_ai)
                print(f"🤖 Agentul a pus piesa în coloana: {col_ai}")

        # Final de meci
        show_board(env.board)
        if env.winner == 1:
            print("\n Ai câștigat!")
        elif env.winner == 2:
            print("\n Agentul a câștigat!")
        else:
            print("\n Remiză!")

if __name__=="__main__":
    agent = QLearningAgent()
    load_agent(agent,"agent.json")
    play_game(agent)