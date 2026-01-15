import pygame
import sys

# -------------------- INIT --------------------
pygame.init()
WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong Game")
CLOCK = pygame.time.Clock()

# -------------------- COLORS --------------------
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# -------------------- GAME OBJECTS --------------------
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
BALL_SIZE = 14

LEFT_PADDLE = pygame.Rect(20, HEIGHT // 2 - PADDLE_HEIGHT // 2,
                          PADDLE_WIDTH, PADDLE_HEIGHT)

RIGHT_PADDLE = pygame.Rect(WIDTH - 30, HEIGHT // 2 - PADDLE_HEIGHT // 2,
                           PADDLE_WIDTH, PADDLE_HEIGHT)

BALL = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2,
                   HEIGHT // 2 - BALL_SIZE // 2,
                   BALL_SIZE, BALL_SIZE)

BALL_SPEED_X = 5
BALL_SPEED_Y = 5
PADDLE_SPEED = 6

# -------------------- FUNCTIONS --------------------
def reset_ball():
    BALL.center = (WIDTH // 2, HEIGHT // 2)
    return -BALL_SPEED_X, BALL_SPEED_Y

# -------------------- MAIN LOOP --------------------
ball_dx, ball_dy = BALL_SPEED_X, BALL_SPEED_Y
running = True

while running:
    # ---- EVENTS ----
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ---- INPUT ----
    keys = pygame.key.get_pressed()

    if keys[pygame.K_w] and LEFT_PADDLE.top > 0:
        LEFT_PADDLE.y -= PADDLE_SPEED
    if keys[pygame.K_s] and LEFT_PADDLE.bottom < HEIGHT:
        LEFT_PADDLE.y += PADDLE_SPEED

    if keys[pygame.K_UP] and RIGHT_PADDLE.top > 0:
        RIGHT_PADDLE.y -= PADDLE_SPEED
    if keys[pygame.K_DOWN] and RIGHT_PADDLE.bottom < HEIGHT:
        RIGHT_PADDLE.y += PADDLE_SPEED

    # ---- BALL MOVEMENT ----
    BALL.x += ball_dx
    BALL.y += ball_dy

    # Wall collision
    if BALL.top <= 0 or BALL.bottom >= HEIGHT:
        ball_dy *= -1

    # Paddle collision
    if BALL.colliderect(LEFT_PADDLE) or BALL.colliderect(RIGHT_PADDLE):
        ball_dx *= -1

    # Out of bounds (reset)
    if BALL.left <= 0 or BALL.right >= WIDTH:
        ball_dx, ball_dy = reset_ball()

    # ---- DRAW ----
    SCREEN.fill(BLACK)
    pygame.draw.rect(SCREEN, WHITE, LEFT_PADDLE)
    pygame.draw.rect(SCREEN, WHITE, RIGHT_PADDLE)
    pygame.draw.ellipse(SCREEN, WHITE, BALL)
    pygame.display.flip()

    CLOCK.tick(60)

pygame.quit()
sys.exit()
