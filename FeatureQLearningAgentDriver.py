import json
import random


class FeatureQLearningAgent:

    def __init__(self, alpha=0.005, gamma=0.9, epsilon=0.2):
        self.alpha = alpha  # Rata de învățare pentru greutăți
        self.gamma = gamma  # Factorul de discount
        self.epsilon = epsilon  # Explorarea

        # Vectorul W: [Bias, Centru, Linii_2_Mele, Linii_3_Mele, Linii_4_Mele, Pericol_3_Opp, Pericol_4_Opp]
        self.weights = [0.0] * 7

    def _get_canonical_board(self, board, my_symbol="A"):
        player_id = 1 if my_symbol == "A" else 2
        if player_id == 1:
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

    def _evaluate_window(self, window, my_id=1, opp_id=2):
        my_c = window.count(my_id)
        opp_c = window.count(opp_id)
        empty_c = window.count(0)

        f_my2 = 1.0 if (my_c == 2 and empty_c == 2) else 0.0
        f_my3 = 1.0 if (my_c == 3 and empty_c == 1) else 0.0
        f_my4 = 1.0 if (my_c == 4) else 0.0
        f_opp3 = 1.0 if (opp_c == 3 and empty_c == 1) else 0.0
        f_opp4 = 1.0 if (opp_c == 4) else 0.0

        return f_my2, f_my3, f_my4, f_opp3, f_opp4

    def extract_features(self, board, action):
        """Transformă o mutare dintr-o stare într-un vector de trăsături numerice phi(s, a)."""
        next_board = self._simulate_move(board, action, player=1)

        f_bias = 1.0
        center_score = (3 - abs(3 - action)) / 3.0  # 1.0 pe centru, 0.0 pe margini

        my2 = my3 = my4 = opp3 = opp4 = 0

        # Orizontal
        for r in range(6):
            for c in range(4):
                m2, m3, m4, o3, o4 = self._evaluate_window(
                    next_board[r][c : c + 4]
                )
                my2 += m2
                my3 += m3
                my4 += m4
                opp3 += o3
                opp4 += o4

        # Vertical
        for c in range(7):
            col_arr = [next_board[r][c] for r in range(6)]
            for r in range(3):
                m2, m3, m4, o3, o4 = self._evaluate_window(col_arr[r : r + 4])
                my2 += m2
                my3 += m3
                my4 += m4
                opp3 += o3
                opp4 += o4

        # Diagonale
        for r in range(3):
            for c in range(4):
                w1 = [next_board[r + i][c + i] for i in range(4)]
                w2 = [next_board[r + 3 - i][c + i] for i in range(4)]
                for w in (w1, w2):
                    m2, m3, m4, o3, o4 = self._evaluate_window(w)
                    my2 += m2
                    my3 += m3
                    my4 += m4
                    opp3 += o3
                    opp4 += o4

        return [f_bias, center_score, my2, my3, my4, opp3, opp4]

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
            if next_valid:
                max_next_q = max(
                    self.get_q_value(next_canonical, a) for a in next_valid
                )
            else:
                max_next_q = 0.0
            target = reward + self.gamma * max_next_q

        # Calculul erorii Temporal Difference (TD-Error)
        td_error = target - current_q

        # Regula Stochastic Gradient Descent pentru actualizarea greutăților
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