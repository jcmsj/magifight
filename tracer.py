import pygame
from saver import save
pygame.init()
win = pygame.display.set_mode((660, 480))  # Adjusted window size
clock = pygame.time.Clock()

GRID_SIZE = 48 
# 48 by 48 grid
grid = []
cell_size = 10

def reset_grid():
    global grid
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
reset_grid()
# Button properties
button_width = 170
button_height = 40
button_margin = 10
buttons = [
    {"label": "Wingardium", "rect": pygame.Rect(485, 10, button_width, button_height), 'cls': 0},
    {"label": "Protego", "rect": pygame.Rect(485, 60, button_width, button_height), 'cls': 1},
    {"label": "Stupefy", "rect": pygame.Rect(485, 110, button_width, button_height), 'cls': 2},
    {"label": "Engorgio", "rect": pygame.Rect(485, 160, button_width, button_height), 'cls': 3},
    {"label": "Reducio", "rect": pygame.Rect(485, 210, button_width, button_height), 'cls': 4},
    {"label": "Reset", "rect": pygame.Rect(485, 260, button_width, button_height), 'cls': 5},
]

font = pygame.font.Font(None, 36)

run = True
is_saving = False  # Flag to track save operation

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.MOUSEBUTTONUP:
            is_saving = False  # Reset flag when left mouse button is released

    win.fill((255, 255, 255))
    
    # draw grid lines
    for i in range(GRID_SIZE+1):
        pygame.draw.line(win, (200, 200, 200), (i * cell_size, 0), (i * cell_size, GRID_SIZE * cell_size))
        pygame.draw.line(win, (200, 200, 200), (0, i * cell_size), (GRID_SIZE * cell_size, i * cell_size))
    
    # draw circles
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i][j] == 1:
                pygame.draw.circle(win, (0, 0, 0), (i * cell_size + cell_size // 2, j * cell_size + cell_size // 2), 5)
    
    # listen to left mouse click
    if pygame.mouse.get_pressed()[0]:
        x, y = pygame.mouse.get_pos()
        i, j = x // cell_size, y // cell_size
        if x < GRID_SIZE * cell_size:  # Only draw circles if not clicking on buttons
            grid[i][j] = 1
        else:
            for button in buttons:
                if button["rect"].collidepoint(x, y) and not is_saving:
                    if button["label"] == "Reset":
                        reset_grid()
                    else:
                        save(grid, button['cls'])
                        reset_grid()
                    is_saving = True  # Set flag to prevent further saves until mouse is released

    keys = pygame.key.get_pressed()
    # listen to enter key, reset grid
    if keys[pygame.K_RETURN]:
        reset_grid()

    # end on escape key
    if keys[pygame.K_ESCAPE]:
        run = False

    # Draw buttons
    for button in buttons:
        pygame.draw.rect(win, (0, 0, 0), button["rect"], 2)
        text_surface = font.render(button["label"], True, (0, 0, 0))
        win.blit(text_surface, (button["rect"].x + 10, button["rect"].y + 5))

    pygame.display.update()
    clock.tick(120)

pygame.quit()
