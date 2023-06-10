import pygame 
import numpy as np
import random

pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
clock = pygame.time.Clock()

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

class Platform:
    def __init__(self):
        self.thickness = 20
        self.top = 500

        self.object = pygame.Rect((
            0, self.top,
            SCREEN_WIDTH, self.thickness
        ))

    def draw(self, color=(150, 150, 150)):
        pygame.draw.rect(SCREEN, color, self.object)

platform = Platform()

run = True
while run:
    clock.tick(60)
    SCREEN.fill((20,20,20))

    key = pygame.key.get_pressed()

    if key[pygame.K_q]:
        run = False

    platform.draw()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()

pygame.quit()

