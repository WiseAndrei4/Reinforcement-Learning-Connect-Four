from GameDriver import GameDriver
import random


class SelfPlayTrainer:
    def __init__(self, agent, random_chance=0.2):
        self.agent = agent
        self.random_chance = random_chance  # 20% chances to match against a random bot

    def train_episode(self):
        env = GameDriver()
        history = {1: None, 2: None}
        
        random_side = random.choice([1, 2]) if random.random() < self.random_chance else None

        while env.winner == -1:
            p = env.step  # 1 sau 2
            opp = 2 if p == 1 else 1

            # Defining the symbols for this episode
            p_symbol = "A" if p == 1 else "B"  # sau ['A', 'B'][p-1]
            opp_symbol = "A" if opp == 1 else "B"

            # 1. ACTUALIZARE AMÂNATĂ (Când rândul revine la jucătorul curent)
            if history[p] is not None:
                prev_state, prev_action = history[p]
                self.agent.learn(
                    prev_state,
                    prev_action,
                    0.0,
                    env.board,
                    done=False,
                    my_symbol=p_symbol,
                )

            # 2. Choosing the action
            state_copy = [row[:] for row in env.board]
            if p == random_side:
                action = random.choice([c for c in range(7) if env.board[0][c] == 0])
            else:
                action = self.agent.choose_action(state_copy, my_symbol=p_symbol)

            # 3. Action execution
            result = env.play(action)

            # 4. Rewards
            if result == "gresit":
                self.agent.learn(
                    state_copy,
                    action,
                    -10.0,
                    env.board,
                    done=True,
                    my_symbol=p_symbol,
                )
                break

            if env.winner == p:
                # The winner evaluates
                self.agent.learn(
                    state_copy,
                    action,
                    1.0,
                    env.board,
                    done=True,
                    my_symbol=p_symbol,
                )

                # The loser evaluates
                if history[opp] is not None:
                    opp_state, opp_action = history[opp]
                    self.agent.learn(
                        opp_state,
                        opp_action,
                        -1.0,
                        env.board,
                        done=True,
                        my_symbol=opp_symbol,
                    )
                break

            elif env.winner == 0:
                # Tie: both receive rewards
                self.agent.learn(
                    state_copy,
                    action,
                    0.5,
                    env.board,
                    done=True,
                    my_symbol=p_symbol,
                )
                if history[opp] is not None:
                    opp_state, opp_action = history[opp]
                    self.agent.learn(
                        opp_state,
                        opp_action,
                        0.5,
                        env.board,
                        done=True,
                        my_symbol=opp_symbol,
                    )
                break

            else:
                # Saving
                history[p] = (state_copy, action)

    def run_training(self, episodes=60000):
        print(f"Pornim antrenamentul pentru {episodes} episoade...")
        decay_rate = (0.01 / self.agent.epsilon) ** (1.0 / episodes)
        for ep in range(episodes):
            self.train_episode()
            self.agent.epsilon = max(0.01, self.agent.epsilon * decay_rate)

    def q_table_size_check(self):
        return self.agent.q_table
