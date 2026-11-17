import json
import random

# Precompute all 69 valid 4-cell windows once for fast feature extraction
WINDOWS = []
for r in range(6):
    for c in range(4):
        WINDOWS.append([(r, c + i) for i in range(4)])
for c in range(7):
    for r in range(3):
        WINDOWS.append([(r + i, c) for i in range(4)])
for r in range(3):
    for c in range(4):
        WINDOWS.append([(r + i, c + i) for i in range(4)])
        WINDOWS.append([(r + 3 - i, c + i) for i in range(4)])


class FeatureQLearningAgent:

    def __init__(self, alpha=0.005, gamma=0.9, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # W: [Bias, Centru, Linii_2_Mele, Linii_3_Mele, Linii_4_Mele, Pericol_2_Opp, Pericol_3_Opp]
        self.weights = [0.0] * 7

    def _get_canonical_board(self, board, my_symbol="A"):
        if my_symbol == "A":
            return board
        return [
            [2 if cell == 1 else (1 if cell == 2 else 0) for cell in row]
            for row in board
        ]

    def _simulate_move(self, board, col, player=1):
        new_board = [row[:] for row in board]
        for r in range(5, -1, -1):
            if new_board[r][col] == 0:
                new_board[r][col] = player
                break
        return new_board

    def get_valid_cols(self, board):
        return [c for c in range(7) if board[0][c] == 0]

    def extract_features(self, board, action):
        """Transformă o mutare într-un vector de trăsături numerice phi(s, a)."""
        next_board = self._simulate_move(board, action, player=1)

        f_bias = 1.0
        center_score = (3 - abs(3 - action)) / 3.0

        my2 = my3 = my4 = opp2 = opp3 = 0.0

        for w in WINDOWS:
            c0 = next_board[w[0][0]][w[0][1]]
            c1 = next_board[w[1][0]][w[1][1]]
            c2 = next_board[w[2][0]][w[2][1]]
            c3 = next_board[w[3][0]][w[3][1]]

            my_c = (c0 == 1) + (c1 == 1) + (c2 == 1) + (c3 == 1)
            opp_c = (c0 == 2) + (c1 == 2) + (c2 == 2) + (c3 == 2)

            if opp_c == 0:
                if my_c == 2:
                    my2 += 1.0
                elif my_c == 3:
                    my3 += 1.0
                elif my_c == 4:
                    my4 += 1.0
            elif my_c == 0:
                if opp_c == 2:
                    opp2 += 1.0
                elif opp_c == 3:
                    opp3 += 1.0

        return [f_bias, center_score, my2, my3, my4, opp2, opp3]

    def get_q_value(self, board, action):
        features = self.extract_features(board, action)
        return sum(w * f for w, f in zip(self.weights, features))

    def choose_action(self, board, my_symbol="A"):
        canonical = self._get_canonical_board(board, my_symbol)
        valid_cols = self.get_valid_cols(board)

        if not valid_cols:
            return 0

        if random.random() < self.epsilon:
            return random.choice(valid_cols)

        q_vals = {col: self.get_q_value(canonical, col) for col in valid_cols}
        max_q = max(q_vals.values())
        best_actions = [col for col, q in q_vals.items() if q == max_q]
        return random.choice(best_actions)

    def learn(self, board, action, reward, next_board, done, my_symbol="A"):
        canonical = self._get_canonical_board(board, my_symbol)
        next_canonical = self._get_canonical_board(next_board, my_symbol)

        features = self.extract_features(canonical, action)
        current_q = sum(w * f for w, f in zip(self.weights, features))

        if done:
            target = reward
        else:
            next_valid = self.get_valid_cols(next_board)
            max_next_q = (
                max(self.get_q_value(next_canonical, a) for a in next_valid)
                if next_valid
                else 0.0
            )
            target = reward + self.gamma * max_next_q

        td_error = max(-5.0, min(5.0, target - current_q))

        for i in range(len(self.weights)):
            self.weights[i] += self.alpha * td_error * features[i]

    def save_weights(self, filename="weights.json"):
        with open(filename, "w") as f:
            json.dump(self.weights, f)

    def load_weights(self, filename="weights.json"):
        try:
            with open(filename, "r") as f:
                self.weights = json.load(f)
        except FileNotFoundError:
            pass