import pygame
from pygame import *
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys
import math
import neat
pygame.init()

size = width, height = (1580,800)

DIR = os.path.dirname(os.path.realpath(__file__))
IMGS_DIR = os.path.join(DIR, "imgs")

win = pygame.display.set_mode(size)
icon = pygame.image.load(os.path.join(IMGS_DIR, "icon.ico"))
pygame.display.set_caption("CARSTAR")

pygame.display.set_icon(icon)

bg_img = pygame.image.load(os.path.join(IMGS_DIR, "circuit.png")).convert()    

    
gen = 0



class Car: 
    def __init__(self):
        self.car_image = pygame.image.load(os.path.join(IMGS_DIR, "car.png"))
        self.car_image = pygame.transform.scale(self.car_image, (50, 50))
        self.rotate_surface = self.car_image

        self.angle = 0
        self.tick = 0
        self.distance = 0
        self.matrix = np.array([20 , height - 200])
        self.center = [self.matrix[0]+ 25, self.matrix[1] + 25]
        self.locations = []
        self.alive = True
        
        
    def check_location(self,obstacle, degree):
        len = 0
        x = int(self.center[0] + np.cos(np.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + np.sin(np.radians(360 - (self.angle + degree))) * len)
        
        while not obstacle.get_at((x, y)) == (230, 230, 230) and len < 300:
            len = len + 1
            x = int(self.center[0] + np.cos(np.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + np.sin(np.radians(360 - (self.angle + degree))) * len)
        #Distance's formula
        dist = int(np.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.locations.append([(x, y), dist])
        
        
        
    def update(self, obstacle):
        self.tick += 1
        self.speed = 15
        self.rotate_surface = self.rot_center(self.car_image,self.angle)
        self.matrix[0] += np.cos(np.radians(360 - self.angle)) * self.speed
        if self.matrix[0] < 20:
            self.matrix[0] = 20
        elif self.matrix[0] > width - 120:
            self.matrix[0] = width - 120
        
        self.distance += self.speed
        
        self.matrix[1] += np.sin(np.radians(360 - self.angle)) * self.speed
        if self.matrix[1] < 20:
            self.matrix[1] = 20
        elif self.matrix[1] > height - 120:
            self.matrix[1] = height - 120
        
        self.center = [int(self.matrix[0]) + 25, int(self.matrix[1]) + 25]
        len = 40
        #The collisions
        
        left_top = [self.center[0] + np.cos(np.radians(360 - (self.angle + 30))) * len, self.center[1] + np.sin(np.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + np.cos(np.radians(360 - (self.angle + 150))) * len, self.center[1] + np.sin(np.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + np.cos(np.radians(360 - (self.angle + 210))) * len, self.center[1] + np.sin(np.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] +  np.cos(np.radians(360 - (self.angle + 330))) * len, self.center[1] + np.sin(np.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]
        
        self.collision(obstacle)
        self.locations.clear()
        for degree in range(-90, 120, 45):
            self.check_location(obstacle,degree)
        
    def draw(self,win):
        win.blit(self.rotate_surface, self.matrix)
        self.draw_location(win)
    def draw_location(self, win):
        for r in self.locations:
            position, dist = r
            pygame.draw.line(win, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(win, (0, 255, 0), position, 5)
    def collision(self, obstacle):
        self.alive = True
        for p in self.four_points:
            if obstacle.get_at((int(p[0]), int(p[1]))) == (230, 230, 230):
                self.alive = False
                break
    def get_alive(self):
        return self.alive
    def get_data(self):
        locations = self.locations
        ret = [0, 0, 0, 0, 0]
        for i, r in enumerate(locations):
            ret[i] = int(r[1] / 30)
        return ret
    def get_reward(self):
        return self.distance / 10.0
    
    def rot_center(self, image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle+ 90)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image


def exit_app():
    pygame.quit()
    sys.exit(0)


def run_AI(genomes, config):
    global  win
    nets = []
    cars = []
    
    for id,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        
        
        cars.append(Car())
    
    
    
    obstacle = bg_img
    clock = pygame.time.Clock()
    global gen
    gen += 1
    
    while True:
        clock.tick(90)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit_app()
        
        
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            i = output.index(max(output))
            if i == 0:
                car.angle += 10
            else:
                car.angle -= 10
                
        alive_cars = 0
        for i, car in enumerate(cars):
            if car.get_alive():
                alive_cars += 1
                car.update(obstacle)
                genomes[i][1].fitness += car.get_reward()    
        if alive_cars == 0:
            break
        win.blit(obstacle, (0,0))
        for car in cars:
            if car.get_alive():
                car.draw(win)
        text = pygame.font.SysFont("arial", 70).render("Generation : " + str(gen), True, (0, 179, 71))
        text_rect = text.get_rect()
        text_rect.center = (width / 2, 20)
        win.blit(text, text_rect)
        
        text = pygame.font.SysFont("arial", 70).render("Alive cars: "+ str(alive_cars), True, (255,0,0))
        text_rect = text.get_rect()
        text_rect.center = (width / 2 , height / 2 - 150)
        win.blit(text, text_rect)
                
        
        pygame.display.flip()
        
        
if __name__ == '__main__':
    config_path = os.path.join(DIR, 'config-feedforward.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.run(run_AI, 1000)

