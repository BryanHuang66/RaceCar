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

class MazeConstructor(tk.Tk):
    def __init__(self,env,path,startpoint,UNIT=16):
        super(MazeConstructor, self).__init__()
        self.env = env
        self.UNIT = UNIT
        self.startpoint = startpoint
        self.MAZE_H = env.race.shape[0]
        self.MAZE_W = env.race.shape[1]
        self.path = path
        self.geometry('{0}x{1}'.format(self.MAZE_H * self.UNIT,
                                       self.MAZE_W * self.UNIT))
        self.tk_build_maze()

    def tk_draw_square(self,x,y,c):
        center = self.UNIT / 2
        w = center - 5
        x_ = self.UNIT * x + center
        y_ = self.UNIT * y + center
        return self.canvas.create_rectangle(x_ - w, y_ - w, x_ + w, y_ + w, fill=c)
    
    def tk_create_line(self,before,after,c):
        center = self.UNIT / 2
        # w = center - 5 
        x1 = before[0] * self.UNIT + center
        y1 = before[1] * self.UNIT + center
        x2 = after[0] * self.UNIT + center
        y2 = after[1] * self.UNIT + center
        return self.canvas.create_line(x1,y1,x2,y2,width=2,arrow=tk.LAST,fill=c)


    def tk_build_maze(self):
        h = self.MAZE_H * self.UNIT
        w = self.MAZE_W * self.UNIT
        # 初始化画布
        self.canvas = tk.Canvas(self, bg='white', height=h, width=w)
        # 画线
        for c in range(0, w, self.UNIT):
            self.canvas.create_line(c, 0, c, h)
        for r in range(0, h, self.UNIT):
            self.canvas.create_line(0, r, w, r)

        # 陷阱
        self.hells = []
        block_index = np.where(self.env.race == 0)
        for i in range(len(block_index[0])):
            self.hells.append(self.tk_draw_square(block_index[1][i], block_index[0][i], 'black'))
        self.hell_coords = []
        for hell in self.hells:
            self.hell_coords.append(self.canvas.coords(hell))
        
        # 奖励
        self.end = []
        oval_index = np.where(self.env.race == 3)
        for i in range(len(oval_index[0])):
            self.end.append(self.tk_draw_square(oval_index[1][i], oval_index[0][i], 'red'))
        self.end_coords = []
        for end in self.end:
            self.end_coords.append(self.canvas.coords(end))

        # 玩家对象
        for item in self.startpoint:
            self.rect = self.tk_draw_square( *item, 'yellow')

        self.canvas.pack()  #执行画
    
    def reset(self,path,startpoint):
        self.canvas.delete("all")
        self.path = path
        self.startpoint = startpoint
        h = self.MAZE_H * self.UNIT
        w = self.MAZE_W * self.UNIT
        # 初始化画布
        # self.canvas = tk.Canvas(self, bg='white', height=h, width=w)
        # 画线
        for c in range(0, w, self.UNIT):
            self.canvas.create_line(c, 0, c, h)
        for r in range(0, h, self.UNIT):
            self.canvas.create_line(0, r, w, r)
        self.hells = []
        block_index = np.where(self.env.race == 0)
        for i in range(len(block_index[0])):
            self.hells.append(self.tk_draw_square(block_index[1][i], block_index[0][i], 'black'))
        self.hell_coords = []
        for hell in self.hells:
            self.hell_coords.append(self.canvas.coords(hell))
        # 路径
        self.pathlist = []
        for i in range(len(self.path)-1):
            self.pathlist.append(self.tk_create_line(self.path[i],self.path[i+1], 'orange'))
        # 奖励
        self.end = []
        oval_index = np.where(self.env.race == 3)
        for i in range(len(oval_index[0])):
            self.end.append(self.tk_draw_square(oval_index[1][i], oval_index[0][i], 'red'))
        self.end_coords = []
        for end in self.end:
            self.end_coords.append(self.canvas.coords(end))
        self.rect = self.tk_draw_square( *self.startpoint, 'yellow')
        self.update()
