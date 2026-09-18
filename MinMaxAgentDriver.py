import math
import random

class MinMaxAgent:

    def __init__(self, depth=6):
        self.depth = depth  # Adâncimea de căutare (6 mutări în avans oferă răspuns instant și joc imbatabil)

    def choose_action(self, board, my_symbol="B"):
        player_id = 1 if my_symbol == "A" else 2
        col, _ = self.minimax(
            board,
            self.depth,
            -math.inf,
            math.inf,
            maximizing=True,
            player_id=player_id,
        )
        return col if col is not None else random.choice(self.get_valid_cols(board))

    def get_valid_cols(self, board):
        return [c for c in range(7) if board[0][c] == 0]

    def make_move(self, board, col, player_id):
        temp_board = [row[:] for row in board]
        for r in range(5, -1, -1):
            if temp_board[r][col] == 0:
                temp_board[r][col] = player_id
                break
        return temp_board

    def check_win(self, board, p):
        # Verificare orizontală, verticală și diagonale
        for r in range(6):
            for c in range(4):
                if all(board[r][c + i] == p for i in range(4)):
                    return True
        for c in range(7):
            for r in range(3):
                if all(board[r + i][c] == p for i in range(4)):
                    return True
        for r in range(3):
            for c in range(4):
                if all(board[r + i][c + i] == p for i in range(4)):
                    return True
        for r in range(3, 6):
            for c in range(4):
                if all(board[r - i][c + i] == p for i in range(4)):
                    return True
        return False

    def evaluate_window(self, window, player_id):
        opp_id = 2 if player_id == 1 else 1
        score = 0
        p_count = window.count(player_id)
        empty_count = window.count(0)
        opp_count = window.count(opp_id)

        if p_count == 4:
            score += 10000
        elif p_count == 3 and empty_count == 1:
            score += 10
        elif p_count == 2 and empty_count == 2:
            score += 2

        if opp_count == 3 and empty_count == 1:
            score -= 80  # Prioritate maximă pe blocarea atacurilor pe margine/centru
        return score

    def score_position(self, board, player_id):
        score = 0
        # Favorizăm coloana din centru (3) pentru că deschide cele mai multe diagonale
        center_array = [board[r][3] for r in range(6)]
        score += center_array.count(player_id) * 4

        # Evaluare ferestre de 4 poziții
        for r in range(6):
            for c in range(4):
                score += self.evaluate_window(board[r][c : c + 4], player_id)
        for c in range(7):
            col_array = [board[r][c] for r in range(6)]
            for r in range(3):
                score += self.evaluate_window(
                    col_array[r : r + 4], player_id
                )
        for r in range(3):
            for c in range(4):
                window = [board[r + i][c + i] for i in range(4)]
                score += self.evaluate_window(window, player_id)
        for r in range(3, 6):
            for c in range(4):
                window = [board[r - i][c + i] for i in range(4)]
                score += self.evaluate_window(window, player_id)
        return score

    def minimax(self, board, depth, alpha, beta, maximizing, player_id):
        opp_id = 2 if player_id == 1 else 1
        valid_cols = self.get_valid_cols(board)

        # Reordonăm coloanele: evaluăm centrul primul pentru retezare masivă (Pruning)
        valid_cols.sort(key=lambda x: abs(3 - x))

        is_win_p = self.check_win(board, player_id)
        is_win_opp = self.check_win(board, opp_id)

        if depth == 0 or is_win_p or is_win_opp or len(valid_cols) == 0:
            if is_win_p:
                return (None, 1000000)
            elif is_win_opp:
                return (None, -1000000)
            elif len(valid_cols) == 0:
                return (None, 0)
            else:
                return (None, self.score_position(board, player_id))

        if maximizing:
            value = -math.inf
            best_col = random.choice(valid_cols)
            for col in valid_cols:
                temp_board = self.make_move(board, col, player_id)
                _, new_score = self.minimax(
                    temp_board, depth - 1, alpha, beta, False, player_id
                )
                if new_score > value:
                    value = new_score
                    best_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # Alpha-Beta Pruning
            return best_col, value
        else:
            value = math.inf
            best_col = random.choice(valid_cols)
            for col in valid_cols:
                temp_board = self.make_move(board, col, opp_id)
                _, new_score = self.minimax(
                    temp_board, depth - 1, alpha, beta, True, player_id
                )
                if new_score < value:
                    value = new_score
                    best_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break  # Alpha-Beta Pruning
            return best_col, value