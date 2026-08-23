import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # Cheie: string-stare, Valoare: lista [q0, q1, ..., q6]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_string(self, board):
        # Grupăm elementele pe coloane, cu '|' pentru a despărți coloanele
        columns = []
        for col in range(7):
            col_str = "".join(str(board[row][col]) for row in range(6))
            columns.append(col_str)
        return "|".join(columns)

    def get_q_values(self, state_str):
        # Dacă starea nu a mai fost întâlnită, o inițializăm cu 0 pentru toate cele 7 acțiuni
        if state_str not in self.q_table:
            self.q_table[state_str] = [0.0] * 7
        return self.q_table[state_str]

    def choose_action(self, board):
        state_str = self.get_state_string(board)
        q_values = self.get_q_values(state_str)

        # Strategia Epsilon-Greedy
        if random.random() < self.epsilon:
            return random.randint(0, 6)  # Explorare: mutare aleatorie
        else:
            # Exploatare: alege acțiunea cu valoarea Q maximă
            max_q = max(q_values)
            # Gestionăm cazul în care sunt mai multe acțiuni cu același scor maxim
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)

    def learn(self, board, action, reward, next_board, done):
        state_str = self.get_state_string(board)
        next_state_str = self.get_state_string(next_board)

        # Extragem valorile Q pentru starea curentă
        q_values = self.get_q_values(state_str)
        current_q = q_values[action]

        # Regula de aur pe care ai observat-o:
        if done:
            max_future_q = 0.0  # Nu mai există mutări viitoare
        else:
            next_q_values = self.get_q_values(next_state_str)
            max_future_q = max(next_q_values)

        # Ecuația Bellman
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        # Actualizăm Q-Table
        self.q_table[state_str][action] = new_q