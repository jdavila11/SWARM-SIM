import math
import sys
import time
from datetime import date
import numpy as np
from matplotlib import pyplot as plt

# Set time parameters
np.set_printoptions(threshold=sys.maxsize)
start = time.time()
today = date.today()

# Experiment parameters
p_step_size = 0.1  # Reduced step size to make agents slower
p_n_agents = 2  # Set to 2 agents with GPS sensors
p_plot = True
p_gps_experiment = True

# Agent parameters
p_ZOR = float(2)
p_ZOA = float(10)
p_s_space = 25
p_sight_range = [p_s_space, 1]
p_gps_error = 4
p_hidden_object_visual_range = 10

# Global variables
g_obstacles = []
g_agents = []
g_hidden_objects = []

class Hidden_Object:
    def __init__(self, x, y):
        self.pos = [float(x), float(y)]
        g_hidden_objects.append(self)
        self.vel = [0, 0]  # Target does not move

    def detect_agents(self):
        return False  # Target does not move, so no need to detect agents

    def move(self):
        pass  # Target does not move

    def stay_in_bounds(self):
        pass  # Target does not move

    def get_distance(self, x, y):
        return math.sqrt((self.pos[0] - x) ** 2 + (self.pos[1] - y) ** 2)

class Agent:
    def __init__(self):
        global g_agents
        g_agents.append(self)
        self.pos = np.zeros(2)
        self.vel = np.random.uniform(-1, 1, 2) * 0.1  # Initial random slower velocity
        self.gps_error = [np.random.normal(0, p_gps_error), np.random.uniform(-1 * np.pi, np.pi)]

    def update_position(self):
        self.pos = np.add(self.pos, self.vel)
        self.stay_in_bounds()

    def detect_target(self, target):
        if self.get_distance(target.pos[0], target.pos[1]) < p_hidden_object_visual_range:
            return True
        return False

    def chase_target(self, target):
        direction = np.subtract(target.pos, self.pos)
        distance = np.linalg.norm(direction)
        if distance > 0:
            self.vel = p_step_size * direction / distance
        else:
            self.vel = [0, 0]

    def get_distance(self, x, y):
        return math.sqrt((self.pos[0] - x) ** 2 + (self.pos[1] - y) ** 2)

    def stay_in_bounds(self):
        if self.pos[0] < -.999 * p_s_space:
            self.pos[0] = -.999 * p_s_space
            self.vel[0] = abs(self.vel[0])  # Reverse velocity direction
        if self.pos[0] > .999 * p_s_space:
            self.pos[0] = .999 * p_s_space
            self.vel[0] = -abs(self.vel[0])  # Reverse velocity direction
        if self.pos[1] < -.999 * p_s_space:
            self.pos[1] = -.999 * p_s_space
            self.vel[1] = abs(self.vel[1])  # Reverse velocity direction
        if self.pos[1] > .999 * p_s_space:
            self.pos[1] = .999 * p_s_space
            self.vel[1] = -abs(self.vel[1])  # Reverse velocity direction

def run_simulation():
    target = Hidden_Object(np.random.uniform(-.75, .9) * p_s_space, np.random.uniform(-.8, .8) * p_s_space)
    agents = [Agent() for _ in range(p_n_agents)]

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    t = 0
    while True:
        for agent in agents:
            if agent.detect_target(target):
                agent.chase_target(target)
            agent.update_position()

        if p_plot and t % 10 == 0:
            plot_positions(t, agents, target, ax)

        t += 1

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot

def plot_positions(t, agents, target, ax):
    ax.clear()
    ax.set_xlim([-1 * p_s_space, p_s_space])
    ax.set_ylim([-1 * p_s_space, p_s_space])
    for agent in agents:
        ax.plot(agent.pos[0], agent.pos[1], 'bo')
    ax.plot(target.pos[0], target.pos[1], 'ro', markersize=12)  # Larger marker size for target
    ax.set_title(f'Time step: {t}')
    plt.pause(0.03)

if __name__ == "__main__":
    run_simulation()






