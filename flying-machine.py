import pygame 
import numpy as np
import random

pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
clock = pygame.time.Clock()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

class Object:
    def __init__(self, prop):
        self.object = pygame.Rect(prop)

        self.cur_pos_x = self.object.x
        self.cur_pos_y = self.object.y

        self.old_pos_x = self.object.x
        self.old_pos_y = self.object.y

        self.acc_x = 0
        self.acc_y = 0

        self.mass = 1

    def getObject(self): 
        return self.object

    def draw(self, color):
        pygame.draw.rect(screen, color , self.object)

    def updatePosition(self):
        temp_x = self.cur_pos_x
        temp_y = self.cur_pos_y

        self.cur_pos_x = self.cur_pos_x * 2 - self.old_pos_x + self.acc_x
        self.cur_pos_y = self.cur_pos_y * 2 - self.old_pos_y + self.acc_y

        self.object.x = int(self.cur_pos_x)
        self.object.y = int(self.cur_pos_y)

        self.old_pos_x = temp_x
        self.old_pos_y = temp_y

        self.acc_x = 0
        self.acc_y = 0

    def applyGravity(self):
        self.acc_y = 0.01

    def applyForce(self, x, y):
        self.acc_x += x / self.mass
        self.acc_y += y / self.mass

player = Object((250,50,100,50))
enemy = Object((
    random.uniform(0, SCREEN_WIDTH), 
    random.uniform(0, SCREEN_HEIGHT),
    50, 50
    ))
platform = Object((0,580, SCREEN_WIDTH, 20))

run = True
while run:
    clock.tick(60)
    screen.fill((20,20,20))

    key = pygame.key.get_pressed()

    f = 0.05
    if key[pygame.K_a]:
        player.applyForce(f, 0)
    if key[pygame.K_d]:
        player.applyForce(-f, 0)
    if key[pygame.K_w]:
        player.applyForce(0, f)
    if key[pygame.K_s]:
        player.applyForce(0, -f)

    if key[pygame.K_q]:
        run = False

    # if not player.getObject().colliderect(platform.getObject()):
    player.updatePosition()
    player.applyGravity()

    player.draw((100,100, 250))
    enemy.draw((250, 100, 100))
    # platform.draw((100, 250, 100))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()

pygame.quit()

