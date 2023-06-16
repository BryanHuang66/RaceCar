import os
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import sys
import time
import pandas as pd
from tqdm import tqdm
import random

class RaceTrackEnv:
    def __init__(self,race_path,max_velocity=5):
        self.race = self.load_race(race_path)
        self.start_position_collect = np.where(self.race == 2)
        self.start_positions = []
        for item in range(len(self.start_position_collect[0])):
            self.start_positions.append((self.start_position_collect[0][item],self.start_position_collect[1][item]))

        self.finish_position_collect = np.where(self.race == 3)
        self.finish_positions = []
        for item in range(len(self.finish_position_collect[0])):
            self.finish_positions.append((self.finish_position_collect[0][item],self.finish_position_collect[1][item]))

        self.max_velocity = max_velocity
        self.reset()

    def reset(self):
        self.position = random.choice(self.start_positions)
        self.velocity = (0, 0)
        return self.position, self.velocity

    def load_race(self,path):
        print('Loading race file...')
        print('Current dir: {}'.format(os.getcwd()))
        print('Race file: {}'.format(path))
        ret = []
        with open(path) as f:
            for line in f:
                ret.append([int(x) for x in line.strip()])
        return np.array(ret, dtype=np.int32)

    def step(self, action):
        new_velocity = (self.velocity[0] + action[0], self.velocity[1] + action[1])
        new_velocity = (max(0, min(new_velocity[0], self.max_velocity)), max(0, min(new_velocity[1], self.max_velocity)))
        if new_velocity == (0, 0):
            new_velocity = (1, 0)

        new_position = (self.position[0] - new_velocity[0], self.position[1] + new_velocity[1])
        self.new_position = new_position

        x1, y1 = self.position
        x2, y2 = self.new_position
        def line_segment_x(x):
            return [(x,round(((y2 - y1) * x + (y1 * x2 - y2 * x1)) / (x2-x1)-0.5)),(x,round(((y2 - y1) * x + (y1 * x2 - y2 * x1)) / (x2-x1)+0.5))]
        def line_segment_y(y):
            return (round(((x2 - x1) * y + (x1 * y2 - x2 * y1)) / (y2-y1)),y)
        
        x_across = list(range(x2,x1))
        y_across = list(range(y1+1,y2+1))

        across_point = []
        for item in x_across:
            across_point += line_segment_x(item)
        for item in y_across:
            across_point.append(line_segment_y(item))
        across_point = list(set(across_point))
        across_point=sorted(across_point,key=lambda t:t[0])
        
        ## 判断是否与终点相交
        for point in across_point:
        # print(new_position)
            ## 终点
            if point[0] in [item[0] for item in self.finish_positions] and point[1] >= self.race.shape[1] - 1 :
                return (new_position, new_velocity), 10, True
            ## 障碍
            elif point[1] > self.race.shape[1] - 1:
                return self.reset(), -50, False
            elif self.race[point] == 0 or point[0] < 0:
                return self.reset(), -50, False
        self.position = new_position
        self.velocity = new_velocity
        return (new_position, new_velocity), -1, False

    def get_actions(self):
        actions = []
        for dv_x in [-1, 0, 1]:
            for dv_y in [-1, 0, 1]:
                actions.append((dv_x, dv_y))
        return actions
    
    def show_race(self):
        plt.imshow(self.race, cmap='gray')
