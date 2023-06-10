import pygame 
import numpy as np
import random

pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
clock = pygame.time.Clock()

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

class Object:
    def __init__(self):
        self.object = None
        self.color = (255,255,255)

        self.position = [0,0]
        self.velocity = [0,0]
        self.acceleration = [0,0]
        self.mass = 1

    def updatePosition(self, dt):
        self.velocity[0] += self.acceleration[0] * dt
        self.velocity[1] += self.acceleration[1] * dt

        self.position[0] += self.velocity[0] * dt + 0.5 * dt * dt * self.acceleration[0]
        self.position[1] += self.velocity[1] * dt + 0.5 * dt * dt * self.acceleration[1]

        self.object.x = int(self.position[0])
        self.object.y = int(self.position[1])

        self.acceleration[0] = 0
        self.acceleration[1] = 0

    def getVelocity(self):
        return [
            self.velocity[0],
            self.velocity[1],
        ]

    def applyVelocity(self, x, y):
        self.velocity[0] += x
        self.velocity[1] += y

    def applyForce(self, x, y):
        self.acceleration[0] += x / self.mass
        self.acceleration[1] += y / self.mass

    def applyGravity(self, x=0, y=9.8):
        self.applyForce(x * self.mass, y * self.mass)

    def setPosition(self, x, y):
        self.position[0] = x
        self.position[1] = y

    def getX(self): 
        return self.position[0]
    def getY(self): 
        return self.position[1]

    def draw(self):
        pygame.draw.rect(SCREEN, self.color, self.object)

class Platform(Object):
    def __init__(self):
        Object.__init__(self)

        self.thickness = 50
        self.top = SCREEN_HEIGHT - self.thickness 
        self.color = (150, 150, 150)

        self.object = pygame.Rect((
            0, self.top,
            SCREEN_WIDTH, self.thickness
        ))

    def checkCollision(self, other):
        if other.object.colliderect(self.object):
            other.setPosition(other.getX(), self.top - other.size)
            v = other.getVelocity()
            other.applyVelocity(0, -v[1])
        else: return None


class Player(Object):
    def __init__(self, x, y):
        Object.__init__(self)

        self.position[0] = x
        self.position[1] = y

        self.size = 50
        self.color = (100, 200, 100)

        self.object = pygame.Rect((
            x, y,
            self.size, self.size
        ))

platform = Platform()
player = Player(200, 100)

run = True
while run:
    clock.tick(60)
    SCREEN.fill((20,20,20))

    key = pygame.key.get_pressed()

    if key[pygame.K_q]:
        run = False

    s = 1
    if key[pygame.K_a]:
        player.applyVelocity(-1 * s, 0)
    elif key[pygame.K_d]:
        player.applyVelocity(1 * s, 0)
    # elif key[pygame.K_w]:
    #     player.applyVelocity(0, -1 * s)
    # elif key[pygame.K_s]:
    #     player.applyVelocity(0, 1 * s)
    else:
        player.applyVelocity(0, 0)

    player.applyGravity()
    player.updatePosition(0.1)

    platform.checkCollision(player)

    platform.draw()
    player.draw()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()

pygame.quit()

