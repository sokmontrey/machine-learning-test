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
        self.name = ""
        self.collision = set()

        self.object = None
        self.color = (255,255,255)

        self.position = [0,0]
        self.velocity = [0,0]
        self.acceleration = [0,0]
        self.mass = 1

        self.damp_factor = 0.0

    def updatePosition(self, dt):
        self.velocity[0] = self.velocity[0] * (1-self.damp_factor) + self.acceleration[0] * dt
        self.velocity[1] = self.velocity[1] * (1-self.damp_factor) +  self.acceleration[1] * dt

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

    def setVelocity(self, x, y):
        self.velocity[0] = 0
        self.velocity[1] = 0

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

    def collide(self, other):
        self.collision.add(other.name)

    def notCollide(self, other):
        if other.name in self.collision:
            self.collision.remove(other.name)

class Platform(Object):
    def __init__(self):
        Object.__init__(self)
        self.name = "ground"

        self.thickness = 50
        self.top = SCREEN_HEIGHT - self.thickness 
        self.color = (150, 150, 150)

        self.object = pygame.Rect((
            0, self.top,
            SCREEN_WIDTH, self.thickness
        ))

        self.friction = 0.1

    def checkCollision(self, other):
        if other.object.colliderect(self.object):
            other.setPosition(other.getX(), self.top - other.size)
            v = other.getVelocity()
            other.applyVelocity(-v[0] * self.friction, -v[1])

            other.collide(self)
        else: 
            other.notCollide(self)


class Player(Object):
    def __init__(self, x, y):
        Object.__init__(self)
        self.name = "player"

        self.position[0] = x
        self.position[1] = y

        self.jump_force = 550 

        self.size = 50
        self.color = (100, 200, 100)

        self.object = pygame.Rect((
            x, y,
            self.size, self.size
        ))

    def jump(self):
        if "ground" in self.collision:
            self.applyForce(0, -1 * self.jump_force)

class Obstacle(Object):
    def __init__(self, x, y):
        Object.__init__(self)
        self.name = "obstacle"

        self.position[0] = x
        self.position[1] = y

        self.size = 50
        self.color = (250, 100, 100)

        self.object = pygame.Rect((
            x, y,
            self.size, self.size
        ))

    def checkCollision(self, other):
        if other.object.colliderect(self.object):
            self.color = (255,255,255)
            other.collide(self)
        else: 
            other.notCollide(self)
            self.color = (250,100,100)

platform = Platform()
player = Player(600, 100)

obstacle = Obstacle(0, 500)

run = True
dt = 0.1
while run:
    clock.tick(60)
    SCREEN.fill((20,20,20))

    key = pygame.key.get_pressed()

    if key[pygame.K_q]:
        run = False

    # s = 1
    # if key[pygame.K_a]:
    #     player.applyVelocity(-1 * s, 0)
    # elif key[pygame.K_d]:
    #     player.applyVelocity(1 * s, 0)
    # elif key[pygame.K_w]:
    #     player.applyVelocity(0, -1 * s)
    # elif key[pygame.K_s]:
    #     player.applyVelocity(0, 1 * s)

    if key[pygame.K_SPACE]:
        player.jump()

    player.applyGravity()
    player.updatePosition(dt)

    if obstacle.getX() > SCREEN_WIDTH:
        obstacle.setPosition(0, obstacle.getY())
    else:
        obstacle.setPosition(obstacle.getX() + 5, obstacle.getY())
    obstacle.updatePosition(dt)

    platform.checkCollision(player)
    obstacle.checkCollision(player)

    platform.draw()
    player.draw()
    obstacle.draw()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()

pygame.quit()

