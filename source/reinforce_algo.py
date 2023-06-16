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

class ReinforcePolicyGradient():
    def __init__(self, env, num_episodes=500, alpha=0.1, gamma=0.99):
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        num_states = (self.env.race.shape[0], self.env.race.shape[1], self.env.max_velocity+1, self.env.max_velocity+1)
        num_actions = len(self.env.get_actions())
        self.theta = np.zeros(num_states + (num_actions,))
        self.episode_rewards = []
        self.episode_lengths = []

    def _map(self, x: tuple):
        return x[0] * 3 + x[1] + 4

    def _remap(self, x: int):
        return ((x-3) // 3, x -4-3*((x-3) // 3))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward(self):
        for _ in tqdm(range(self.num_episodes), position=0, leave=True):
            state = self.env.reset()
            done = False
            episode = []
            episode_reward = 0
            episode_length = 0

            while not done:
                if random.random() < 0.1:
                    action = 4
                else:
                    action_probs = self.softmax(self.theta[state[0] + state[1]])
                    action = np.random.choice(len(action_probs), p=action_probs)
                action = self._remap(action)
                next_state, reward, done = self.env.step(action)
                # print(state, action, next_state, reward)
                episode.append((state, action, reward))
                episode_reward += reward
                episode_length += 1
                state = next_state

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            G = 0
            for t in range(len(episode) - 1, -1, -1):
                state_t, action_t, reward_t = episode[t]
                G = self.gamma * G + reward_t
                self.theta[state_t[0] + state_t[1] + (self._map(action_t),)] += self.alpha * G


    def predict(self, state=-1, limit=1000):
        if state == -1:
            state = self.env.reset()
        policy = []
        for _ in tqdm(range(limit)):
            policy.append(state)
            if random.random() < 0.1:
                action = 4
            else:
                action_probs = self.softmax(self.theta[state[0] + state[1]])
                action = np.argmax(action_probs)
            action = self._remap(action)
            next_state, _, done = self.env.step(action)
            state = next_state
            if done:
                policy.append(next_state)
                break
        return policy


class SarsaLambda():
    def __init__(self,env,num_episodes=500,alpha=0.1,gamma=0.99,epsilon=0.1,lam=0.9):
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam
        self.env = env
        num_states = (self.env.race.shape[0], self.env.race.shape[1],self.env.max_velocity+1,self.env.max_velocity+1)  
        num_actions = len(self.env.get_actions())
        self.q_values = np.zeros(num_states + (num_actions,))
        self.eligibility_trace = np.zeros_like(self.q_values)
        self.episode_rewards = []

    def _map(self,x:tuple):
        return x[0]*3+x[1]+4

    
    def epsilon_greedy_policy(self, state, q_values, epsilon, actions):
        if random.random() < 0.1:
            return (0,0)
        else:
            if random.random() < epsilon:
                return random.choice(actions)
            else:
                return actions[np.argmax(q_values[state[0] + state[1]])]
        
    def forward(self):
        for _ in tqdm(range(self.num_episodes),position=0, leave=True):
            state = self.env.reset()
            action = self.epsilon_greedy_policy(state, self.q_values, self.epsilon, self.env.get_actions())
            episode_reward = 0
            while True:
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                if done:
                    self.episode_rewards.append(episode_reward)
                    break
                next_action = self.epsilon_greedy_policy(next_state, self.q_values, self.epsilon, self.env.get_actions())
                delta = reward + self.gamma * self.q_values[next_state[0] + next_state[1] + (self._map(next_action),)] - self.q_values[state[0] + state[1] + (self._map(action),)]
                self.eligibility_trace[state[0]+state[1] + (self._map(action),)] += 1

                self.q_values += self.alpha * delta * self.eligibility_trace
                self.eligibility_trace *= self.gamma * self.lam

                state = next_state
                action = next_action
    
    def predict(self,state=-1,limit=1000):
        if state == -1:
            state = self.env.reset()
        policy = []
        for _ in tqdm(range(limit)):
            policy.append(state)
            action = self.epsilon_greedy_policy(state, self.q_values, -1, self.env.get_actions())
            next_state, _, done = self.env.step(action)
            state = next_state
            if done:
                policy.append(next_state)
                break
        return policy
