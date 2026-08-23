from GameDriver import GameDriver
import random

class SelfPlayTrainer:
    def __init__(self, agent, random_chance=0.2):
        self.agent = agent
        self.random_chance = random_chance  # 20% șanse de meci împotriva unui bot random

    def train_episode(self):
        env = GameDriver()
        history = {1: None, 2: None}

        # Decidem tipul meciului pentru acest episod
        is_random_match = random.random() < self.random_chance

        while env.winner == -1:
            p = env.step  # Jucătorul curent (1 sau 2)
            opp = 2 if p == 1 else 1

            # 1. Actualizare amânată (doar pentru entitățile care învață)
            # Botul random (Jucătorul 2 în meciurile random) nu are Q-table și nu învață
            if p == 1 or (p == 2 and not is_random_match):
                if history[p] is not None:
                    prev_state, prev_action = history[p]
                    self.agent.learn(
                        prev_state, prev_action, 0.0, env.board, done=False
                    )

            state_copy = [row[:] for row in env.board]

            # 2. Alegerea acțiunii
            if p == 2 and is_random_match:
                # Oponentul Random: alege o coloană la întâmplare
                action = random.randint(0, 6)
            else:
                # Agentul Inteligent (Self-Play sau când e Jucătorul 1)
                action = self.agent.choose_action(state_copy)

            # 3. Executarea mutării
            result = env.play(action)

            # 4. Gestionarea penalizărilor și recompenselor finale
            if result == "gresit":
                if p == 1 or (p == 2 and not is_random_match):
                    self.agent.learn(
                        state_copy, action, -10.0, env.board, done=True
                    )
                break  # Mutare invalidă

            if env.winner == p:
                # Câștigătorul primește +1 (dacă e entitate care învață)
                if p == 1 or (p == 2 and not is_random_match):
                    self.agent.learn(
                        state_copy, action, 1.0, env.board, done=True
                    )
                # Pierzătorul primește -1 (dacă învață)
                if history[opp] is not None and (
                    opp == 1 or (opp == 2 and not is_random_match)
                ):
                    opp_state, opp_action = history[opp]
                    self.agent.learn(
                        opp_state, opp_action, -1.0, env.board, done=True
                    )
                break

            elif env.winner == 0:
                # Remiză: ambii primesc +0.5
                if p == 1 or (p == 2 and not is_random_match):
                    self.agent.learn(
                        state_copy, action, 0.5, env.board, done=True
                    )
                if history[opp] is not None and (
                    opp == 1 or (opp == 2 and not is_random_match)
                ):
                    opp_state, opp_action = history[opp]
                    self.agent.learn(
                        opp_state, opp_action, 0.5, env.board, done=True
                    )
                break

            else:
                # Jocul continuă. Salvăm starea doar dacă jucătorul învață
                if p == 1 or (p == 2 and not is_random_match):
                    history[p] = (state_copy, action)

    def run_training(self, episodes=60000):
        print(f"Pornim antrenamentul hibrid pentru {episodes} episoade...")
        for ep in range(episodes):
            self.train_episode()

            # Scădem explorarea treptat la fiecare 2000 de meciuri
            if ep % 2000 == 0 and ep > 0:
                self.agent.epsilon = max(0.01, self.agent.epsilon * 0.95)

        print(
            f"Antrenament complet! Q-Table-ul are acum {len(self.agent.q_table)} stări memorate."
        )

    def q_table_size_check(self):
        return self.agent.q_table