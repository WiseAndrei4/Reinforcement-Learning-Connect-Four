import random
import json

class QTableAgent:

    def __init__(
        self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1
    ):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_canonical_board(self, board, my_symbol="A"):
        """Aduce tabla la forma standard: piesele MELE devin 'A', ale ADVERSARULUI devin 'B'."""
        if my_symbol == "A":
            return board

        canonical_board = []
        for row in board:
            new_row = []
            for cell in row:
                if cell == 1:
                    new_row.append(2)
                elif cell == 2:
                    new_row.append(1)
                else:
                    new_row.append(0)
            canonical_board.append(new_row)
        return canonical_board

    def get_state_string(self, board, my_symbol="A"):
        # Normalizează mai întâi tabla în funcție de simbolul agentului
        canonical = self.get_canonical_board(board, my_symbol)

        columns = []
        for col in range(7):
            col_str = "".join(str(canonical[row][col]) for row in range(6))
            columns.append(col_str)
        return "|".join(columns)

    def get_q_values(self, state_str):
        if state_str not in self.q_table:
            self.q_table[state_str] = [0.0] * 7
        return self.q_table[state_str]

    def choose_action(self, board, my_symbol="A"):
        state_str = self.get_state_string(board, my_symbol)
        q_values = self.get_q_values(state_str)

        # Identificăm doar coloanele valide (care nu sunt pline)
        valid_cols = [c for c in range(7) if board[0][c] == 0]

        # Dacă coloanele valide sunt pline (caz extrem), fallback pe 0..6
        if not valid_cols:
            valid_cols = list(range(7))

        # 1. Explorare (Epsilon-Greedy)
        if random.random() < self.epsilon:
            return random.choice(valid_cols)

        # 2. Exploatare: dacă starea este nouă (toate valorile Q sunt 0.0)
        if all(q == 0.0 for q in q_values):
            # Alege o coloană validă aleatorie (evită blocarea pe coloana 0)
            return random.choice(valid_cols)

        # 3. Alege acțiunea cu Q maxim DINTRE COLOANELE VALIDE
        valid_q = {col: q_values[col] for col in valid_cols}
        max_q = max(valid_q.values())
        best_actions = [col for col, q in valid_q.items() if q == max_q]

        return random.choice(best_actions)

    def learn(
        self,
        board,
        action,
        reward,
        next_board,
        done,
        my_symbol="A",
    ):
        state_str = self.get_state_string(board, my_symbol)
        next_state_str = self.get_state_string(next_board, my_symbol)

        current_q = self.get_q_values(state_str)[action]

        if done:
            max_future_q = 0.0
        else:
            max_future_q = max(self.get_q_values(next_state_str))

        # Ecuația Bellman
        new_q = current_q + self.alpha * (
            reward + self.gamma * max_future_q - current_q
        )
        self.q_table[state_str][action] = new_q

    def load_agent(agent, filename="agent.json"):
        try:
            with open(filename, "r") as f:
                agent.q_table = json.load(f)
                print("Succesfully loaded!")
        except FileNotFoundError:  # Corectat din FileExistsError
            print("Missing file! Playing with empty Q-table.")
