import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
"""
maze = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
])

start = (0, 0)
goal = (9, 9)

state       yukarı      aşağı  sol  sağ
a1           0          0       0   0
a2          0           0       0   0
a3          0           0       0   0
yeni bilgi = eski bilgi + öğrenme oranı x (şimdiki ödül + gelecekteki en iyi tahmin - eski bilgi)




R = np.array([
    [-1,-1,-1,-1,0,-1],
    [-1,-1,-1,0,-1,100],
    [-1,-1,-1,-1,0,-1],
    [-1,0,0,0,-1,-1],
    [0,-1,-1,0,-1,100],
    [-1,0,-1,-1,0,100],
])

Q = np.zeros((6,6))
gamma = 0.8 #geleceğe önemi
alpha = 0.9 #öğrenme oranı

for i in range(1000):
    s = np.random.randint(0,6)
    possible_actions = np.where(R[s] >=0)[0]
    a = np.random.choice(possible_actions)
    s_next = a
    Q[s,a] = Q[s,a] + alpha * (R[s,a] + gamma * Q[s_next].max() - Q[s,a])

print("q tablosu",np.round(Q,2))
print("*"*100)

lab = np.array([
    [0,0,10],
    [0,0,0],
    [0,0,0]
])
Q = np.zeros((3,3,4))
action = [0,1,2,3] # 0= yukarı, 1=aşağı, 2=sola, 3 =sağ
alpha = 0.8
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

def get_next_state(i,j,action):
    if action == 0 and i > 0: i -= 1
    elif action == 1 and i < 2: i += 1
    elif action == 2 and j > 0: j -= 1
    elif action == 3 and j < 2: j += 1
    return i,j

def get_reward(i,j):
    return lab[i,j] -1
for episode in range(500):
    i,j = 0,0
    for step in range(20):
        if random.uniform(0,1) < epsilon:
            a = random.choice(action)
        else:
            a = np.argmax(Q[i,j])
        ni,nj = get_next_state(i,j,a)
        reward = get_reward(ni,nj)
        Q[i,j,a] += alpha * (reward + gamma*np.max(Q[ni,nj]) - Q[i,j,a])
        if lab[ni,nj] == 10:
            break
        i,j = ni,nj

print(np.round(Q,2))
# 0= yukarı, 1=aşağı, 2=sola, 3 =sağ
mapping = {0:"^", 1:"-", 2: "<", 3:">"}
goal = (0,2)

print("-"*100)
for i in range(3):
    row = []
    for j in range(3):
        if (i,j) == goal:
            row.append("G")
        else:
            a_best = np.argmax(Q[i,j])
            row.append(mapping[a_best])
    print(''.join(row))
"""

#[0] [1] [2] [3] [4]

states = [0,1,2,3,4]
actions = [0,1] #0 : sol
goal_state = 4

def step(state, action):
    if action == 1: #sağ
        new_state = state + 1
    else:
        new_state = state -1

    if new_state<0:
        return 0, -5, True #duvara çarptı
    elif new_state == goal_state:
        return goal_state, 10, True #hedefe ulaştı
    else:
        return new_state, -1, False #normal adım attı

Q = np.zeros((len(states), len(actions)))
print(Q)

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
episodes = 100000

for ep in range(episodes):
    state = 0 #her bölüm başta sıfırdan başlaması gerekiyor...
    done = False
    while not done:
        if random.uniform(0,1) < epsilon:
            action = random.choice(actions) #keşif
        else:
            action = np.argmax(Q[state]) #sömürü

        new_state, reward, done = step(state,action)

        old_value = Q[state,action]
        next_max = np.max(Q[new_state])
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        Q[state,action] = new_value

        state = new_state
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
print("-"*10,"q","-"*10)
print(np.round(Q,2))

state = 0
path = [state]

while state != goal_state:
    action = np.argmax(Q[state])
    new_state, reward, done = step(state,action)
    path.append(new_state)
    state = new_state
    if done:
        break
print("öğrendiği yol ?", path)











































