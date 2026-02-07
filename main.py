import pygame
import random
import numpy as np
import sys

# --- НАЛАШТУВАННЯ ---
SCREEN_SIZE = 600
GRID_COUNT = 20
CELL_SIZE = SCREEN_SIZE // GRID_COUNT
FPS = 15

# Колірна гама (Emerald & Gold)
CLR_BG = (5, 15, 10)  # Темно-зелена ніч
CLR_SNAKE_HEAD = (16, 185, 129)  # Emerald 500
CLR_SNAKE_BODY = (5, 150, 105)  # Emerald 600
CLR_FOOD = (251, 191, 36)  # Amber 400 (Золото)
CLR_GRID = (10, 30, 20)  # Ледь помітна сітка
CLR_UI_BG = (2, 10, 5)  # Чорний ліс
CLR_TEXT = (209, 213, 219)  # Світло-сірий
CLR_BTN_ACTIVE = (16, 185, 129)
CLR_BTN_IDLE = (31, 41, 55)
CLR_BTN_RESET = (153, 27, 27)  # Red 800 (для скидання)
CLR_BTN_RESET_HOVER = (185, 28, 28)

# RL Параметри
LEARNING_RATE = 0.1
DISCOUNT = 0.95


class SnakeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.head = [GRID_COUNT // 2, GRID_COUNT // 2]
        self.snake = [list(self.head), [self.head[0], self.head[1] + 1], [self.head[0], self.head[1] + 2]]
        self.direction = 0  # 0: Вгору, 1: Вниз, 2: Вліво, 3: Вправо
        self.food = self._place_food()
        self.score = 0
        self.steps_since_food = 0
        return self._get_state()

    def _place_food(self):
        while True:
            food = [random.randint(0, GRID_COUNT - 1), random.randint(0, GRID_COUNT - 1)]
            if food not in self.snake:
                return food

    def _get_state(self):
        state = [
            self.food[1] < self.head[1],  # їжа вище
            self.food[1] > self.head[1],  # їжа нижче
            self.food[0] < self.head[0],  # їжа лівіше
            self.food[0] > self.head[0],  # їжа правіше
            self.head[1] == 0 or [self.head[0], self.head[1] - 1] in self.snake,
            self.head[1] == GRID_COUNT - 1 or [self.head[0], self.head[1] + 1] in self.snake,
            self.head[0] == 0 or [self.head[0] - 1, self.head[1]] in self.snake,
            self.head[0] == GRID_COUNT - 1 or [self.head[0] + 1, self.head[1]] in self.snake
        ]
        return tuple(state)

    def step(self, action):
        if action == 0 and self.direction != 1:
            self.direction = 0
        elif action == 1 and self.direction != 0:
            self.direction = 1
        elif action == 2 and self.direction != 3:
            self.direction = 2
        elif action == 3 and self.direction != 2:
            self.direction = 3

        new_head = list(self.head)
        if self.direction == 0:
            new_head[1] -= 1
        elif self.direction == 1:
            new_head[1] += 1
        elif self.direction == 2:
            new_head[0] -= 1
        elif self.direction == 3:
            new_head[0] += 1

        reward = 0
        done = False

        if (new_head[0] < 0 or new_head[0] >= GRID_COUNT or
                new_head[1] < 0 or new_head[1] >= GRID_COUNT or
                new_head in self.snake):
            reward = -100
            done = True
            return self._get_state(), reward, done

        self.head = new_head
        self.snake.insert(0, list(self.head))
        self.steps_since_food += 1

        if self.head == self.food:
            reward = 50
            self.score += 1
            self.food = self._place_food()
            self.steps_since_food = 0
        else:
            self.snake.pop()
            dist_old = abs(self.snake[1][0] - self.food[0]) + abs(self.snake[1][1] - self.food[1])
            dist_new = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
            reward = 1 if dist_new < dist_old else -1

        if self.steps_since_food > 150:
            reward = -20
            done = True

        return self._get_state(), reward, done


class Brain:
    def __init__(self):
        self.q_table = {}

    def reset_brain(self):
        self.q_table = {}

    def get_action(self, state, mode):
        # mode: 'raw' (випадково), 'ai' (навчений)
        if mode == 'raw':
            return random.randint(0, 3)
        return np.argmax(self._check_state(state))

    def _check_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        return self.q_table[state]

    def learn(self, state, action, reward, next_state):
        old_q = self._check_state(state)[action]
        next_max = np.max(self._check_state(next_state))
        self.q_table[state][action] = (1 - LEARNING_RATE) * old_q + LEARNING_RATE * (reward + DISCOUNT * next_max)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE + 120))
    pygame.display.set_caption("Нейро-Змійка: Еволюція")
    clock = pygame.time.Clock()
    font_main = pygame.font.SysFont("Verdana", 16, bold=True)
    font_sub = pygame.font.SysFont("Verdana", 11)

    env = SnakeEnv()
    ai = Brain()

    mode = 'raw'
    is_training_background = False
    episodes_total = 1
    max_score = 0
    total_food_eaten = 0

    # Прямокутники кнопок
    btn_raw = pygame.Rect(20, SCREEN_SIZE + 70, 110, 35)
    btn_train = pygame.Rect(140, SCREEN_SIZE + 70, 180, 35)
    btn_ai = pygame.Rect(330, SCREEN_SIZE + 70, 100, 35)
    btn_reset = pygame.Rect(440, SCREEN_SIZE + 70, 140, 35)

    while True:
        state = env.reset()
        done = False

        while not done:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_raw.collidepoint(event.pos):
                        mode = 'raw'
                    elif btn_ai.collidepoint(event.pos):
                        mode = 'ai'
                    elif btn_train.collidepoint(event.pos):
                        is_training_background = True
                    elif btn_reset.collidepoint(event.pos):
                        # Скидання всього
                        ai.reset_brain()
                        episodes_total = 1
                        max_score = 0
                        total_food_eaten = 0
                        mode = 'raw'
                        state = env.reset()

            if is_training_background:
                for _ in range(500):
                    t_state = env.reset()
                    t_done = False
                    while not t_done:
                        t_action = random.randint(0, 3) if random.random() < 0.2 else np.argmax(
                            ai._check_state(t_state))
                        t_n_state, t_reward, t_done = env.step(t_action)
                        ai.learn(t_state, t_action, t_reward, t_n_state)
                        t_state = t_n_state
                        if t_reward == 50: total_food_eaten += 1
                is_training_background = False
                episodes_total += 500
                mode = 'ai'
                state = env.reset()

            action = ai.get_action(state, mode)
            next_state, reward, done = env.step(action)

            if reward == 50: total_food_eaten += 1
            ai.learn(state, action, reward, next_state)
            state = next_state

            # --- ВІЗУАЛІЗАЦІЯ ---
            screen.fill(CLR_BG)

            # Сітка
            for x in range(0, SCREEN_SIZE, CELL_SIZE):
                pygame.draw.line(screen, CLR_GRID, (x, 0), (x, SCREEN_SIZE))
            for y in range(0, SCREEN_SIZE, CELL_SIZE):
                pygame.draw.line(screen, CLR_GRID, (0, y), (SCREEN_SIZE, y))

            # Їжа
            f_rect = (env.food[0] * CELL_SIZE + 4, env.food[1] * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8)
            pygame.draw.rect(screen, CLR_FOOD, f_rect, border_radius=CELL_SIZE // 2)

            # Змійка
            for i, part in enumerate(env.snake):
                color = CLR_SNAKE_HEAD if i == 0 else CLR_SNAKE_BODY
                p_rect = (part[0] * CELL_SIZE + 2, part[1] * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4)
                pygame.draw.rect(screen, color, p_rect, border_radius=6)

            # UI Панель
            pygame.draw.rect(screen, CLR_UI_BG, (0, SCREEN_SIZE, SCREEN_SIZE, 120))

            # Малювання кнопок
            pygame.draw.rect(screen, CLR_BTN_ACTIVE if mode == 'raw' else CLR_BTN_IDLE, btn_raw, border_radius=5)
            pygame.draw.rect(screen, CLR_BTN_IDLE, btn_train, border_radius=5)
            pygame.draw.rect(screen, CLR_BTN_ACTIVE if mode == 'ai' else CLR_BTN_IDLE, btn_ai, border_radius=5)

            # Кнопка скидання з ефектом наведення
            curr_reset_clr = CLR_BTN_RESET_HOVER if btn_reset.collidepoint(mouse_pos) else CLR_BTN_RESET
            pygame.draw.rect(screen, curr_reset_clr, btn_reset, border_radius=5)

            screen.blit(font_sub.render("ВИПАДКОВО", True, CLR_TEXT), (btn_raw.x + 15, btn_raw.y + 10))
            screen.blit(font_sub.render("ТРЕНУВАТИ (500 ІГОР)", True, CLR_TEXT), (btn_train.x + 15, btn_train.y + 10))
            screen.blit(font_sub.render("РЕЖИМ ШІ", True, CLR_TEXT), (btn_ai.x + 18, btn_ai.y + 10))
            screen.blit(font_sub.render("ОЧИСТИТИ ВСЕ", True, CLR_TEXT), (btn_reset.x + 22, btn_reset.y + 10))

            # Розширена статистика
            max_score = max(max_score, env.score)
            status_color = CLR_SNAKE_HEAD if mode == 'ai' else CLR_FOOD
            mode_text = "СТАН: ІНТЕЛЕКТ АКТИВНИЙ" if mode == 'ai' else "СТАН: ХАОТИЧНИЙ РУХ"

            lbl_mode = font_sub.render(mode_text, True, status_color)
            lbl_score = font_main.render(f"РАХУНОК: {env.score:02} | РЕКОРД: {max_score}", True, CLR_TEXT)

            # Додаткові метрики
            col2_x = SCREEN_SIZE - 230
            lbl_episodes = font_sub.render(f"ЕПОХИ (ІГРИ): {episodes_total}", True, CLR_TEXT)
            lbl_states = font_sub.render(f"БАЗА ЗНАНЬ: {len(ai.q_table)} СТАНІВ", True, CLR_TEXT)
            lbl_food = font_sub.render(f"ЗДОБИЧ: {total_food_eaten} ОД.", True, CLR_TEXT)

            screen.blit(lbl_mode, (20, SCREEN_SIZE + 10))
            screen.blit(lbl_score, (20, SCREEN_SIZE + 32))

            screen.blit(lbl_episodes, (col2_x, SCREEN_SIZE + 10))
            screen.blit(lbl_states, (col2_x, SCREEN_SIZE + 28))
            screen.blit(lbl_food, (col2_x, SCREEN_SIZE + 46))

            pygame.display.flip()
            clock.tick(FPS)

        episodes_total += 1


if __name__ == "__main__":
    main()