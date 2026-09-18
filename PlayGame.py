import sys
import time
import pygame

from FeatureQLearningAgentDriver import FeatureQLearningAgent
from MinMaxAgentDriver import MinMaxAgent

from GameDriver import GameDriver

# Configurații Vizuale
ROWS, COLS = 6, 7
SQUARESIZE = 100
WIDTH = COLS * SQUARESIZE
HEIGHT = (ROWS + 1) * SQUARESIZE  # Rândul 0 e rezervat pentru status și preview
RADIUS = int(SQUARESIZE / 2 - 8)

# Paletă de Culori Modernă
COLOR_BG = (15, 23, 42)  # Dark Navy Background
COLOR_BOARD = (30, 58, 138)  # Royal Blue Grid
COLOR_EMPTY = (15, 23, 42)
COLOR_HUMAN = (239, 68, 68)  # Roșu
COLOR_AI = (234, 179, 8)  # Galben
COLOR_TEXT = (241, 245, 249)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Connect Four AI - Live Stand Demo")
font_status = pygame.font.SysFont("Arial", 22, bold=True)


def draw_board(board):
    """Randează grila Connect Four și piesele existente."""
    for r in range(ROWS):
        for c in range(COLS):
            # Căsuța albastră
            pygame.draw.rect(
                screen,
                COLOR_BOARD,
                (c * SQUARESIZE, (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE),
                border_radius=4,
            )

            # Cerc gol / piesă
            cell = board[r][c]
            color = COLOR_EMPTY
            if cell == 1:
                color = COLOR_HUMAN
            elif cell == 2:
                color = COLOR_AI

            pygame.draw.circle(
                screen,
                color,
                (
                    int(c * SQUARESIZE + SQUARESIZE / 2),
                    int((r + 1) * SQUARESIZE + SQUARESIZE / 2),
                ),
                RADIUS,
            )


def main():
    agent = FeatureQLearningAgent()
    agent.load_weights("weights.json")
    agent.epsilon = 0.0  # Fără explorare aleatorie în meciul cu jucător uman

    env = GameDriver()
    game_over = False
    message = "Rândul tău! Click pe coloană pentru a muta."

    clock = pygame.time.Clock()

    while True:
        screen.fill(COLOR_BG)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Mutare Om (Click Stânga)
            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and not game_over
                and env.step == 1
            ):
                posx = event.pos[0]
                col = posx // SQUARESIZE

                res = env.play(col)
                if res != "gresit":
                    if env.winner == 1:
                        message = "🏆 Ai câștigat! Apasă [R] pentru meci nou."
                        game_over = True
                    elif env.winner == 0:
                        message = "🤝 Remiză! Apasă [R] pentru meci nou."
                        game_over = True

            # Reîncepere joc (Tasta R)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env = GameDriver()
                game_over = False
                message = "Rândul tău! Click pe coloană pentru a muta."

        # Rândul AI-ului (Automatizat)
        if not game_over and env.step == 2:
            # Afișăm temporar că AI-ul gândește
            pygame.draw.rect(screen, COLOR_BG, (0, 0, WIDTH, SQUARESIZE))
            txt = font_status.render("🤖 AI-ul se gândește...", True, COLOR_TEXT)
            screen.blit(txt, (20, 35))
            draw_board(env.board)
            pygame.display.update()

            time.sleep(0.35)  # Pauză mică pentru efect vizual la stand

            state_copy = [row[:] for row in env.board]
            col_ai = agent.choose_action(state_copy, my_symbol="B")

            res = env.play(col_ai)
            while res == "gresit":
                col_ai = (col_ai + 1) % 7
                res = env.play(col_ai)

            if env.winner == 2:
                message = "🤖 AI-ul a câștigat! Apasă [R] pentru meci nou."
                game_over = True
            elif env.winner == 0:
                message = "🤝 Remiză! Apasă [R] pentru meci nou."
                game_over = True

        # Preview piesa de sus când omul plimbă mouse-ul
        if not game_over and env.step == 1:
            posx = pygame.mouse.get_pos()[0]
            col = posx // SQUARESIZE
            pygame.draw.circle(
                screen,
                COLOR_HUMAN,
                (int(col * SQUARESIZE + SQUARESIZE / 2), int(SQUARESIZE / 2)),
                RADIUS,
            )

        # Randăm textul de status și tabla
        txt_surface = font_status.render(message, True, COLOR_TEXT)
        screen.blit(txt_surface, (20, 10 if env.step == 2 else 35))
        draw_board(env.board)

        pygame.display.update()
        clock.tick(60)


if __name__ == "__main__":
    main()