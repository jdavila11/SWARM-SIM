# change to the rules - agents have to go back and recharge when they 'run out of battery'
# working on GPS error and plotting the actual vs estimated position
# target will run away now
# swarm needs to detect the target earlier and then chase it
import csv
import math
import operator
import os
import sys
import time
from datetime import date
import numpy as np
import pandas as pd
import cProfile
from matplotlib import pyplot as plt
from numpy.random.mtrand import normal
# to remove gps error, make all agents accurate.
# to remove target moving, comment out in demo.
# to remove endurance, add a few zeros to p_endurance
# set time parameters
np.set_printoptions(threshold=sys.maxsize)
start = time.time()
today = date.today()
# set experiment parameters
p_detection_range = [0, 1, 1, 1, 1, 1, 1, 1, 1, 3]  # lo 4 agents, hi 4 single
p_target_change_frequency = 1
p_iterations = 1500  # number of steps allowed before giving up
p_endurance = 1000000*1000  # number of calculations before a recharge is needed (1 mil for small area, 10 m for large area)
runs = 900  # update for number of runs
#np.random.seed(35)
p_step_size = 10  # actual step size is one divided by this number
p_n_agents = 9
p_central_axis = 0
p_plot = True
p_gps_experiment = False
p_wind = False
p_show_history = False
p_coverage_experiment = False
p_distance_experiment = False
p_alt_frequency = 500
p_s_space = 25
p_pheromone_decay_rate = 0.4
p_pheromone_emission_rate = 1
p_labels_no_mem = ['Baseline Collection of Individuals (NOT A SWARM)', 'Basic Swarm', 'Swarm Using Line Strategy',
                   'Flanking Circle', 'Flanking Triangle', 'Flanking Platoon', 'Flying-V',
                   'Platoon Column Movements', 'Single Agent Lawnmower Search']
p_labels_mem = ['Baseline Collection of Individuals w/Memory (NOT A SWARM)', 'Basic Swarm w/ Memory',
                'Swarm Using Line Strategy w/Memory', 'Flanking Circle w/Memory', 'Flanking Triangle w/Memory',
                'Flanking Platoon w/Memory', 'Flying-V w/Memory', 'Platoon Column Movements w/Memrory',
                'Single Agent Lawnmower Search w/Memory']
'''p_resolution = [0, int(1.5*p_s_space), int(1.5*p_s_space), int(1.5*p_s_space), int(1.5*p_s_space), int(1.5*p_s_space),
                int(1.5*p_s_space), int(1.5*p_s_space), int(1.5*p_s_space), int(0.5*p_s_space), int(0.5*p_s_space),
                int(0.5*p_s_space), int(0.5*p_s_space), int(0.5*p_s_space), int(0.5*p_s_space), int(0.5*p_s_space),
                int(0.5*p_s_space), int(0.5*p_s_space)]'''  # how many pixels in each dimension
p_random_target_modifications_x = np.random.uniform(0, .45, runs)
p_random_target_modifications_y = np.random.uniform(-.95, .95, runs)
# set agent parameters
p_ZOR = float(2)
p_ZOR_tri = float(3.2)
p_ZOA = float(10)
p_swarm_desires = [.1, .025, .85, .025]  # [alignment, attraction, separation, cohesion] each as a percentage
p_wander = 0.0  # ****** modify this parameter for experimentation on the best probability of leaving the swarm ******
p_wander_frequency = 100
p_line_width = p_n_agents * p_ZOR * 0.8  # width of the line formed by all the agents, minus 20% for overlap
p_time_lag = 0  # loops that will pass before an agent receives another agent's position
p_blind_spot = [0, 0]  # radians [average, standard dev]
if p_gps_experiment:
    p_gps_error = 4  # standard deviation for original gps error
    p_gps_autocor_mag = .5
    p_gps_autocor_dir = .1
else:
    p_gps_error = 0
    p_gps_autocor_mag = 0
    p_gps_autocor_dir = 0
p_sight_range = [p_s_space, 1]  # [average, standard dev]
p_avoid_ang_l = [1, 99, 5, 95, 9, 91, 13, 87, 17, 83, 21, 79, 25, 75, 29, 71, 67, 33, 63, 37, 59, 41, 55, 45, 51, 49]
p_avoid_ang_r = [51, 49, 55, 45, 57, 43, 61, 39, 65, 35, 69, 31, 73, 27, 77, 23, 81, 19, 85, 15, 89, 11, 93, 7, 97, 3]
p_flanking_velocities = [[0, 0], [1, 0], [0, -1], [-1, 0], [0, 1]]
p_opposite_flanks = [0, 3, 4, 1, 2]
p_flank_radars = [0, 0, 25, 50, 75]
p_hidden_object_visual_range = 10

# declare global variables
g_obstacles = []
g_pixels = []
g_agents = []
g_hidden_objects = []
g_percent_distance = 100
g_pixels_explored = 0
g_agent_distances = []
g_found_target = False
g_gps_error = 0
g_charger_available = True
g_charging_spots = [[-.9*p_s_space, -.9*p_s_space], [-.9*p_s_space, -.8*p_s_space], [-.8*p_s_space, -.9*p_s_space],
                    [-.9*p_s_space, -.7*p_s_space], [-.7*p_s_space, -.9*p_s_space], [-.9*p_s_space, -.6*p_s_space],
                    [-.6*p_s_space, -.9*p_s_space], [-.5*p_s_space, -.9*p_s_space], [-.9*p_s_space, -.5*p_s_space]]

if not p_coverage_experiment and not p_distance_experiment:
    g_data = np.zeros((runs, 1))  # this array will show the number of steps to reach target
else:
    g_data = np.zeros((runs, 10001))  # this array will show distance or coverage vs step number


def start_drag():  # draggable wind function
    os.system('python Draggable.py')


class Obstacle:
    def __init__(self):
        self.pos = [np.random.uniform(-.5 * p_s_space, -.2 * p_s_space),
                    np.random.uniform(-.9 * p_s_space, -.7 * p_s_space)]
        # self.pos[1] = -20  # delete
        for o in g_obstacles:
            if o.pos[1] - .6*p_s_space < self.pos[1] < o.pos[1] + .6*p_s_space:
                self.pos[1] += 1.2 * p_s_space
        self.size = np.random.uniform(9, 10)  # radius
        g_obstacles.append(self)


class Hidden_Object:
    def __init__(self, x, y):
        self.pos = [float(x), float(y)]
        g_hidden_objects.append(self)
        self.visible_agents = []
        self.vel = [0, 0]
        self.wander_direction = [0, 0]

    def detect_agents(self, c):
        c = min(c, 4)
        self.visible_agents.clear()
        for agent in g_agents:
            d = math.sqrt((agent.pos[c][0] - self.pos[0]) ** 2 + (agent.pos[c][1] - self.pos[1]) ** 2)
            if d < p_hidden_object_visual_range:
                self.visible_agents.append(agent)

    def get_distance2(self, w, z, x, y):  # a function that returns the distance between two points
        d = math.sqrt((w-x)**2 + (z-y)**2)
        return d

    def run_away(self, c):
        c = min(c, 4)
        self.vel = [0, 0]
        for agent in self.visible_agents:
            self.vel += np.subtract(self.pos, agent.pos[c])

    def move_around(self, q):
        c = min(q, 4)
        '''if q % 200 == 0:
            self.wander_direction = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        self.vel = np.dot(p_step_size, self.wander_direction)'''
        self.vel = [0, 0]
        dead_reckoning = np.add(self.vel, self.pos)  # true pos
        for agent in self.visible_agents:  # avoid collisions
            # dead reckon the agent's position
            # if agent will hit me, move to the side
            dead_reckoning = np.add(agent.vel, agent.pos[c])
            if self.get_distance2(dead_reckoning[0], dead_reckoning[1], self.pos[0], self.pos[1]) < 3*p_ZOR:
                bearing = math.atan2(agent.pos[c][1]-self.pos[1], agent.pos[c][0]-self.pos[0])
                option1 = bearing + np.pi/2
                if option1 < -np.pi:
                    option1 += 2*np.pi
                if option1 > np.pi:
                    option1 -= 2*np.pi
                option2 = bearing - np.pi/2
                self.vel = [p_step_size*math.cos(option1), p_step_size*math.sin(option1)]
                #self.vel = [p_step_size * math.cos(heading - np.pi / 2), p_step_size * math.sin(heading - np.pi / 2)]


    def stay_in_bounds(self):
        if self.pos[0] < -.999 * p_s_space:
            self.vel[0] = max(self.vel[0], 0)
        if self.pos[0] > .999 * p_s_space:
            self.vel[0] = min(self.vel[0], 0)
        if self.pos[1] < -.999 * p_s_space:
            self.vel[1] = max(self.vel[1], 0)
        if self.pos[1] > .999 * p_s_space:
            self.vel[1] = min(self.vel[1], 0)

    def normalize_vel(self):
        velocity_magnitude = max((p_step_size * math.sqrt(self.vel[0] ** 2 + self.vel[1] ** 2)), 1)
        self.vel = np.divide(self.vel, velocity_magnitude)

    def update_pose(self):
        self.pos = np.add(self.pos, self.vel)


class Wind:
    def __init__(self):
        self.force = float(0)
        self.direction = 0.5 * np.pi
        self.components = [self.force * math.cos(self.direction), self.force * math.sin(self.direction)]

    def change(self):
        linear_change = np.random.normal(loc=0.0, scale=0.1)
        angular_change = np.random.normal(loc=0.0, scale=0.05)
        self.force += linear_change
        if self.force < 0:
            self.force = 0
            self.direction += angular_change
        if self.direction > np.pi:
            self.direction -= 2 * np.pi
        if self.direction < -np.pi:
            self.direction += 2 * np.pi


def get_xy(filename):
    magnitude = 0
    direction = 0
    with open(filename) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        for row in csv_reader:
            magnitude = row[0]
            direction = row[1]
    return magnitude, direction


class Pixel:
    def __init__(self, row, column, res):
        left_bound = (-1 * p_s_space) + (column * p_s_space * 2 / res)
        right_bound = left_bound + p_s_space * 2 / res
        bottom_bound = (-1 * p_s_space) + (row * p_s_space * 2 / res)
        top_bound = bottom_bound + p_s_space * 2 / res
        self.centerx = (left_bound + right_bound) / 2
        self.centery = (top_bound + bottom_bound) / 2
        self.pheromones = float(0)
        self.explored = False
        g_pixels.append(self)

    def freshen_up(self, u):
        if u > 0:
            self.pheromones -= p_pheromone_decay_rate  # decay rate every 10 loops
        self.pheromones = max(self.pheromones, 0)

    def stink_up(self, u):
        global g_pixels_explored
        if not self.explored:
            self.explored = True
            g_pixels_explored += 1
        if u > 0:
            self.pheromones += p_pheromone_emission_rate


class Agent:
    def __init__(self):
        global g_agents
        self.tracking = False
        self.column = 1
        self.column_leader = False
        self.reset_q = 0
        self.saved_progress = [0, 0]
        self.need_resume = False
        self.accurate = False
        self.charging = False
        self.need_charge = False
        self.line_width = p_line_width
        self.calculations = 0
        self.rank = 0
        self.parent_rank = 1
        self.left_wing = False
        self.name = 0
        self.gps_error = [np.random.normal(0, p_gps_error), np.random.uniform(-1 * np.pi, np.pi)]  # [range, bearing]
        if len(g_agents) == 0:
            self.leadership = 1
        else:
            self.leadership = np.random.uniform(0, 1, 1)
        self.best_leader = self.leadership
        self.destination = None
        self.orig_y = 0  # for avoiding obstacles. go back to original y after passing the obstacle.
        self.radar = 100*np.zeros(100)
        self.need_return = False
        self.near_obstacle = False
        self.my_column_nbrs = []
        self.last_backout = 0
        # already described in word doc from here down
        g_agents.append(self)  # update the global list that keeps track of all the agents
        self.avoid_obstacle = False
        self.behavior = 0  # 0 or 1. 0 means follow the swarm, 1 means wander off.
        self.blind_spot = np.random.normal(loc=p_blind_spot[0], scale=p_blind_spot[1], size=1)
        self.color = 'b'  # blue
        self.flanking = 1  # flank sequence. a=central axis(c.a), b=(c.a) + pi/2, c =(c.a) + pi, d=(c.a) + 3pi/2
        self.pos = np.zeros((5, 2), float)  # to avoid holding too much data, only store most recent 5
        self.pos[0][0] = -.97*p_s_space
        self.pos[0][1] = -1*p_s_space + p_ZOR*1.5*(len(g_agents)-1)
        self.orig_y = self.pos[0][1]
        for agent in g_agents:  # make sure agents initialize a decent distance from one another
            if math.sqrt((self.pos[0][0] - agent.pos[0][0]) ** 2 + (self.pos[0][1] - agent.pos[0][1]) ** 2) < p_ZOR:
                self.pos[0][0] += .1 * p_s_space
        self.heading = 0
        self.north_end = False
        self.est_pos = np.zeros(2, float)  # estimated position
        self.NNL = self
        self.NNR = self
        self.north_south = 0  # 0 is north 1 is south
        self.pivot_point = float(-100)  # a reference point for mowing the lawn, how much to move up for new pass
        self.south_end = False
        self.vel = [0, 0]
        self.v_nbrs_left = []
        self.v_nbrs_right = []
        self.v_nbrs_all = []
        self.v_nbrs_close = []
        self.visual_range = np.random.normal(loc=p_sight_range[0], scale=p_sight_range[1], size=1)
        self.wander_direction = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]

    def detect_target(self, c, target):
        if self.get_distance(c, target.pos[0], target.pos[1]) < .00001*p_hidden_object_visual_range:
            self.tracking = True

    def chase(self, c, target):
        self.vel = np.subtract(target.pos, self.pos[c])

    def rtb_recharge(self, c, q, formation):
        global g_charger_available
        if self.get_distance(c, -.9 * p_s_space, -.9 * p_s_space) < 15:
            if self.need_charge:
                g_charger_available = False
                self.charging = True
                if self.get_distance(c, -.9 * p_s_space, -.9 * p_s_space) < .1:
                    self.vel = [0, 0]
                self.calc_inc(-1000)
                if formation == 7:
                    for n in self.v_nbrs_all:
                        n.calculations = min(self.calculations, n.calculations)  # everyone in the formation recharges
                        if n.calculations < 0:
                            n.need_charge = False
        if self.calculations < 0:
            self.need_charge = False
            self.charging = False
            g_charger_available = True
        if self.need_charge and self.get_distance(c, -.9 * p_s_space, -.9 * p_s_space) >= 0.1:
            self.vel = np.subtract([-.9 * p_s_space, -.9 * p_s_space], self.est_pos)  # head back to recharge station
            if self.near_obstacle:  # dodge obstacles on the way
                self.make_safe()
        if formation >= 3:
            if not self.need_charge and self.need_resume:
                self.vel = np.subtract(self.saved_progress, self.est_pos)  # head back to where I left off in search
                if self.near_obstacle:  # dodge obstacles on the way
                    self.make_safe()
                if self.get_distance(c, self.saved_progress[0], self.saved_progress[1]) <= 0.1:
                    self.need_resume = False
                if 4 <= formation <= 6:
                    leader = self.check_leader(q)
                    if leader != self:
                        self.need_resume = False
                if formation == 8:
                    leader = self.check_next_up(q)
                    if leader != self:
                        self.need_resume = False

    def calc_inc(self, n):
        self.calculations += n

    # functions used for all formations
    def can_see_neighbor_left_right(self, c, neighbor):  # determine whether I can see a neighbor with each eye
        can_see = [False, False]  # [left right]
        neighbor.distance = self.get_distance(c, neighbor.pos[c][0], neighbor.pos[c][1])  # true position
        if neighbor.distance <= self.visual_range and neighbor != self:
            bearing = math.atan2(neighbor.pos[c][1] - self.pos[c][1], neighbor.pos[c][0] - self.pos[c][0])  # true pos
            if bearing < 0:
                bearing += 2 * np.pi
            if 0 <= bearing - self.heading <= np.pi - self.blind_spot / 2:  # 2
                can_see[0] = True
            elif bearing - self.heading <= -np.pi + self.blind_spot / 2:  # 3
                can_see[0] = True
            if 0 <= self.heading - bearing <= np.pi - self.blind_spot / 2:  # 4
                can_see[1] = True
            elif self.heading - bearing <= -np.pi + self.blind_spot / 2:  # 5
                can_see[1] = True
        self.calc_inc(5)
        return can_see

    def check_laser(self, c, rng):  # high range for circle, platoon flank, etc
        self.radar = np.zeros(100)  # true bearings, pi/50 increments [0, pi/50, 2pi/50, ... , 98pi/50, 99pi/50]
        self.near_obstacle = False
        for a in range(100):  # for every angle  # 1
            for o in g_obstacles:
                d = self.check_obstacle(self.pos[c], np.array([np.cos(a*np.pi/50), np.sin(a*np.pi/50)]), o.pos, o.size)
                if 0 < d < p_ZOR + rng:
                    self.radar[a] = 1
                    self.near_obstacle = True
        self.calc_inc(200)

    def check_obstacle(self, p, d, c, r):  # p=agent pos, d=ray direction, c=circle center, r=radius
        # normalize the direction vector
        d = d / np.linalg.norm(d)

        # calculate the vector from the center of the circle to the endpoint of the ray
        pc = p - c

        # calculate the discriminant of the quadratic equation
        discriminant = np.dot(pc, d) ** 2 - np.dot(pc, pc) + r ** 2

        # check if the ray intersects the circle
        if discriminant < 0:
            distance = 0
        else:
            # calculate the distances from the endpoint of the ray to the locations where the ray intersects the circle
            sqrt_discriminant = np.sqrt(discriminant)
            distances = np.dot(-1, [np.dot(pc, d) - sqrt_discriminant, np.dot(pc, d) + sqrt_discriminant])
            if max(distances) < 0:
                distance = 0
            else:
                distance = min(distances)
        return distance

    def cycle_positions(self, t):
        self.calc_inc(5)
        if t < 4:
            self.pos[t + 1] = np.add(self.pos[t], self.vel)
        else:
            for w in range(4):
                self.pos[w] = self.pos[w + 1]
            self.pos[4] = np.add(self.pos[4], self.vel)

    def error_change(self):
        if self.accurate:
            self.gps_error = [0, 0]
        else:
            new_magnitude = np.random.normal(p_gps_autocor_mag*self.gps_error[0], p_gps_error)
            new_direction = (self.gps_error[1] + np.random.normal(0, p_gps_autocor_dir)) % (2 * np.pi)
            self.gps_error = [new_magnitude, new_direction]

    def get_distance(self, c, x, y):  # a function that returns the distance to a point
        self.calc_inc(1)
        d = math.sqrt((self.pos[c][0] - x) ** 2 + (self.pos[c][1] - y) ** 2)
        return d

    def get_distance2(self, w, z, x, y):  # a function that returns the distance between two points
        self.calc_inc(1)
        d = math.sqrt((w-x)**2 + (z-y)**2)
        return d

    def heading_match_velocity(self):
        angular_speed = 0.1 * (math.atan2(self.vel[1], self.vel[0]) - self.heading)
        self.heading = (self.heading + angular_speed) % (2 * np.pi)
        if self.heading < 0:
            self.heading += 2 * np.pi

    def list_visible_agents(self, c, s):  # s is specialty - extra lists for certain functions, 1-line, 2-column
        self.v_nbrs_all.clear()  # start fresh each loop. may add memory later  # 1
        self.v_nbrs_close.clear()  # 2
        if s == 1:
            self.v_nbrs_left.clear()
            self.v_nbrs_right.clear()
            self.calc_inc(2)
        elif s == 2:
            self.my_column_nbrs.clear()
            self.calc_inc(1)
        for agent in g_agents:
            if agent != self:
                if max(self.can_see_neighbor_left_right(c, agent)) == 1:  # If can see w/ either eye # 3
                    self.v_nbrs_all.append(agent)  # 4
                    if agent.distance < 1.1*p_ZOR_tri:  # 5
                        self.v_nbrs_close.append(agent)  # 6
                    if s == 2:  # column
                        if agent.column == self.column:
                            self.my_column_nbrs.append(agent)
                    self.calc_inc(3)
                if s == 1:  # line
                    if self.can_see_neighbor_left_right(c, agent)[0] and agent != self:
                        self.v_nbrs_left.append(agent)
                    if self.can_see_neighbor_left_right(c, agent)[1] and agent != self:
                        self.v_nbrs_right.append(agent)
                    self.calc_inc(2)
        self.calc_inc(6)

    def normalize_velocity(self):
        velocity_magnitude = max((p_step_size * math.sqrt(self.vel[0] ** 2 + self.vel[1] ** 2)), 1)
        self.vel = np.divide(self.vel, velocity_magnitude)  # normalize to a step size of 0.1

    def rcv_gps(self, c):
        self.calc_inc(1)
        self.est_pos = [self.pos[c][0] + self.gps_error[0] * math.cos(self.gps_error[1]),
                            self.pos[c][1] + self.gps_error[0] * math.sin(self.gps_error[1])]

    def dgps(self, leader, c):
        projected = np.add(self.est_pos, np.subtract(leader.pos[c], self.pos[c]))  # where agent sees leader
        differential = np.subtract(leader.pos[c], projected)
        self.est_pos = np.add(self.est_pos, differential)
        self.gps_error[0] = np.sqrt((self.pos[c][0]-self.est_pos[0])**2+(self.pos[c][1]-self.est_pos[1])**2)
        self.gps_error[1] = math.atan2(self.pos[c][1]-self.est_pos[1], self.pos[c][0]-self.est_pos[0])

    def dgps2(self, c):  # if close to the recharge station, can dgps off of that.
        if self.pos[c][0] < -.89*p_s_space and self.pos[c][1] < -.89*p_s_space:
            self.est_pos = self.pos[c]
            self.gps_error = [0, 0]

    def update_data(self, c_pose, target):
        global g_percent_distance
        global g_agent_distances
        global g_gps_error
        distance_to_target = self.get_distance(c_pose, target.pos[0], target.pos[1])
        g_agent_distances.append(distance_to_target)
        if distance_to_target < g_percent_distance:
            g_percent_distance = distance_to_target
        g_gps_error += self.gps_error[0]

    def make_safe(self):
        a = int(math.atan2(self.vel[1], self.vel[0])//(np.pi/50))
        if max(self.radar[a], self.radar[a-1], self.radar[a+1]) == 1:
            new_angle = None
            for j in range(99):
                if a - j < -99:
                    j -= 100
                if self.radar[a-j] == 0:
                    new_angle = (a-j)*np.pi/50
                    break
                if a + j > 99:
                    j -= 100
                if self.radar[a+j] == 0:
                    new_angle = (a+j)*np.pi/50
                    break
            if new_angle is None:
                self.vel = [0, 0]
            else:
                self.vel = [p_step_size * math.cos(new_angle), p_step_size * math.sin(new_angle)]
            self.calc_inc(4*j + 2)

    def mark_pixel(self, q, c, res):
        x = max(min(self.est_pos[0], p_s_space), -1 * p_s_space)  # est pos
        y = max(min(self.est_pos[1], p_s_space), -1 * p_s_space)  # est pos
        my_pixel = int(((y + p_s_space) // (p_s_space * 2 / res)) * res + ((x + p_s_space) // (p_s_space * 2 / res)))
        if 0 <= my_pixel < len(g_pixels):
            g_pixels[my_pixel].stink_up(q)
        self.calc_inc(5)
        return my_pixel

    def memory_adjustment(self, my_pixel, res, strength, c, snake):  # snake t/f. only needs memory for above and below
        self.calc_inc(1)
        if abs(self.est_pos[1]) < .8*p_s_space:  # make sure I'm not on the top or bottom edge
            self.calc_inc(1)
            if abs(self.est_pos[0]) < .8*p_s_space:  # make sure I'm not on the left or right edge
                self.calc_inc(12)
                if snake:
                    surndig_pixls = [my_pixel - 5*res, my_pixel + 5*res]
                    pix_ajustmnts = [[0, -4*strength], [0, 4*strength]]
                else:
                    surndig_pixls = [my_pixel - 1, my_pixel - res - 1, my_pixel - res, my_pixel - res + 1,
                                     my_pixel + 1, my_pixel + res + 1, my_pixel + res, my_pixel + res - 1]
                    pix_ajustmnts = [[-4*strength, 0], [-2*strength, -2*strength], [0, -4*strength],
                                     [2*strength, -2*strength], [4*strength, 0], [2*strength, 2*strength], [0, 4*strength],
                                     [-2*strength, -2*strength]]
                for p in surndig_pixls:
                    if g_pixels[p].pheromones == 0:
                        self.vel = np.add(self.vel, np.multiply(p_step_size, pix_ajustmnts[surndig_pixls.index(p)]))

    # functions used for some but not all formations:
    def snake(self, c, v, pivot_length, strategy, cone_width):  # strategy refers to flank=1, column=2 or flying V=3
        self.calc_inc(5)
        if self.rank != 1:
            self.calc_inc(1)
            self.vel = p_flanking_velocities[self.flanking]
        if self.flanking == 1 or self.flanking == 3:
            self.calc_inc(2)
            self.reset_north_south(self.north_south, self.north_south, self.north_south)
            if strategy == 1:
                self.snake_flank(c, v, p_flank_radars[self.flanking], cone_width)
            else:
                self.snake_turn(c, v, p_flank_radars[self.flanking])
        if (self.flanking == 1 and self.est_pos[0] > p_s_space + pivot_length) or \
                (self.flanking == 3 and self.est_pos[0] < -p_s_space - pivot_length):
            self.calc_inc(4)
            if self.north_south == 0:
                self.flanking = 4
                self.reset_q = 0
                self.avoid_obstacle = False
            else:
                self.flanking = 2
                self.reset_q = 0
                self.avoid_obstacle = False
            if strategy != 3:
                self.set_pivot_point(4)
        if self.flanking % 2 == 0:
            self.calc_inc(3)
            if self.flanking == 4:
                self.calc_inc(2)
                if self.est_pos[1] > p_s_space - 1.5 * pivot_length:
                    self.north_south = 1
                    self.calc_inc(1)
                if self.est_pos[1] > self.pivot_point + 1.25 * pivot_length:
                    self.calc_inc(2)
                    if self.est_pos[0] > 0:
                        self.vel = [-.5 * v * p_step_size, .5 * v * p_step_size]
                    else:
                        self.vel = [.5 * v * p_step_size, .5 * v * p_step_size]
            if self.flanking == 2:
                self.calc_inc(2)
                if self.est_pos[1] < 1.5 * pivot_length - p_s_space:
                    self.calc_inc(1)
                    self.north_south = 0
                if self.est_pos[1] < self.pivot_point - 1.25 * pivot_length:
                    self.calc_inc(2)
                    if self.est_pos[0] > 0:
                        self.vel = [-.5 * v * p_step_size, -.5 * v * p_step_size]
                    else:
                        self.vel = [.5 * v * p_step_size, -.5 * v * p_step_size]
            if (self.flanking == 2 and self.est_pos[1] < self.pivot_point - 2.5 * pivot_length) or \
                    (self.flanking == 4 and self.est_pos[1] > self.pivot_point + 2.5 * pivot_length):
                self.calc_inc(3)
                if self.est_pos[0] > 0:
                    self.flanking = 3
                    self.orig_y = self.est_pos[1]
                else:
                    self.flanking = 1
                    self.orig_y = self.est_pos[1]

    def snake_flank(self, c, v, a, s):  # c-pose, v-vel, a-flank dir, s = cone width (big 4 shapes, small 4 slawnmower)
        self.avoid_obstacle = False
        self.calc_inc(1)
        for t in range(s):
            if self.radar[a - t] == 1 or self.radar[a+t] == 1:
                self.avoid_obstacle = True
                break
        self.calc_inc(t)
        if not self.avoid_obstacle:
            if a == 0:
                if self.orig_y - .05 < self.est_pos[1] < self.orig_y + .05:  # snake right
                    self.vel = [v * p_step_size, 0]
                    self.calc_inc(3)
                else:
                    if not self.near_obstacle:
                        self.vel = [v * p_step_size, v * (self.orig_y - self.est_pos[1])]  # return to orig trajectory
                        self.calc_inc(3)
            else:
                if self.orig_y - .05 < self.est_pos[1] < self.orig_y + .05:  # snake left
                    self.vel = [-1 * v * p_step_size, 0]
                    self.calc_inc(3)
                else:
                    if not self.near_obstacle:
                        self.calc_inc(3)
                        self.vel = [-1 * v * p_step_size, v * (self.orig_y - self.est_pos[1])]  # return to trajectory
        else:  # find new angle
            self.calc_inc(3)
            new_angle = 0
            if a == 0:
                avoid_angles = p_avoid_ang_l
            else:
                avoid_angles = p_avoid_ang_r
            for t in avoid_angles:
                if self.radar[t] == 0:
                    new_angle = t*np.pi/50
                    break
            self.calc_inc(t)
            self.vel = [v/2 * p_step_size * math.cos(new_angle), v/2 * p_step_size * math.sin(new_angle)]

    def snake_backout(self, c, q, lb):
        self.calc_inc(1)
        if self.get_distance(c, self.pos[0][0], self.pos[0][1]) < .1 and q-lb > 5:
            self.calc_inc(3)
            reverse_flanks = [0, 3, 4, 1, 2]
            self.orig_y -= 1
            self.last_backout = q
            self.flanking = reverse_flanks[self.flanking]

    def flank(self, v, d):  # v is velocity, d is direction (1 east, 2 south, 3 west, 4 north)
        self.vel = np.dot(p_flanking_velocities[d], v * p_step_size)
        self.calc_inc(1)

    def clear_ahead(self, a, cone):  # checking radar within a few angles (cone) each side of the heading (a)
        self.calc_inc(4)
        if self.avoid_obstacle and max(self.radar) == 0:
            self.avoid_obstacle = False
            self.calc_inc(1)
        if cone == 1:
            if max(self.radar[a], self.radar[a+1], self.radar[a-1]) == 0:
                return True
            else:
                return False
        else:
            if max(max(self.radar[a:a+cone]), max(self.radar[a-cone:a-1]), self.radar[a]) == 0:
                return True
            else:
                self.avoid_obstacle = True
                return False

    def snake_turn(self, c, v, a):  # c-current pose, v-vel, a-direction of flanking motion (0 or 50)
        self.calc_inc(2)
        if self.need_return and max(max(self.radar[a+1:a+30]), self.radar[a], max(self.radar[a-30:a-1])) == 0:
            if self.est_pos[1] < self.orig_y - p_ZOR:
                self.flank(v, 4)
            elif self.est_pos[1] > self.orig_y + p_ZOR:
                self.flank(v, 2)
            else:
                self.need_return = False
        else:
            if self.clear_ahead(a, cone=7):
                if a == 0:
                    self.flank(v, 1)
                else:
                    self.flank(v, 3)
            else:
                if not self.need_return:
                    self.need_return = True
                    self.orig_y = self.est_pos[1]
                if self.north_south == 0:
                    self.flank(v, 4)
                else:
                    self.flank(v, 2)

    def snake_leader(self, c, v, pivot_length):
        self.calc_inc(5)
        if self.flanking % 2 == 0:  # if going up or down
            if self.est_pos[0] < 0:
                if self.est_pos[0] > -1*p_s_space + .5*pivot_length:
                    if self.flanking == 2:
                        self.vel = [-.5 * v * p_step_size, -.5 * v * p_step_size]
                    else:
                        self.vel = [-.5 * v * p_step_size, .5 * v * p_step_size]
                else:
                    self.vel = p_flanking_velocities[self.flanking]
            else:
                if self.est_pos[0] < p_s_space - .5*pivot_length:
                    if self.flanking == 2:
                        self.vel = [.5 * v * p_step_size, -.5 * v * p_step_size]
                    else:
                        self.vel = [.5 * v * p_step_size, .5 * v * p_step_size]
                else:
                    self.vel = p_flanking_velocities[self.flanking]

    def stay_in_bounds(self):
        self.calc_inc(4)
        if self.est_pos[0] < -.999 * p_s_space:
            self.vel[0] = max(self.vel[0], 0)
        if self.est_pos[0] > .999 * p_s_space:
            self.vel[0] = min(self.vel[0], 0)
        if self.est_pos[1] < -.999 * p_s_space:
            self.vel[1] = max(self.vel[1], 0)
        if self.est_pos[1] > .999 * p_s_space:
            self.vel[1] = min(self.vel[1], 0)

    # specialty functions
    def calculate_spot_in_formation(self, c):  # Line only. ID nearest l/r neighbor, determine whether I'm an end
        self.calc_inc(5)
        self.north_end = self.south_end = False  # assume I am not an end
        self.NNR = self.NNL = self  # until I find my nearest neighbor right and left, I am my own NNR/NNL
        if not self.v_nbrs_left:  # if I can't see anyone on my left, I must be the left end.
            self.north_end = True
        else:  # if I can see agents on my left, the closest one to me is my NNL
            for agent in self.v_nbrs_left:
                agent.range = self.get_distance(c, agent.pos[c][0], agent.pos[c][1])
            sorted_left_neighbors = sorted(self.v_nbrs_left, key=operator.attrgetter("range"))
            if not sorted_left_neighbors[0].need_charge and not sorted_left_neighbors[0].need_return:
                self.NNL = sorted_left_neighbors[0]
        if not self.v_nbrs_right:
            self.south_end = True
        else:
            for agent in self.v_nbrs_right:
                agent.range = self.get_distance(c, agent.pos[c][0], agent.pos[c][1])
            sorted_right_neighbors = sorted(self.v_nbrs_right, key=operator.attrgetter("range"))
            if not sorted_right_neighbors[0].need_charge and not sorted_right_neighbors[0].need_return:
                self.NNR = sorted_right_neighbors[0]
        line_south_end = self.est_pos[1]
        line_north_end = self.est_pos[1]
        for agent in self.v_nbrs_right:  # estimate the line width
            line_south_end = min(line_south_end, agent.pos[c][1])
        for agent in self.v_nbrs_left:
            line_north_end = max(line_north_end, agent.pos[c][1])
        self.line_width = line_north_end - line_south_end

    def change_behavior(self):
        self.calc_inc(2)
        if self.behavior == 1:
            self.behavior = 0
        else:
            self.behavior = 1

    def check_shape_spots(self, c, ldr, shape):  # find an acceptable spot relative to the leader
        self.calc_inc(10)
        if shape == 'circle':
            crds = [np.add(np.add(ldr.vel, ldr.pos[c]), [-.696*p_ZOR, 3.94*p_ZOR]),
                    np.add(np.add(ldr.vel, ldr.pos[c]), [.696*p_ZOR, -3.94*p_ZOR]),
                    np.add(np.add(ldr.vel, ldr.pos[c]), [3.94*p_ZOR, .696*p_ZOR]),
                    np.add(np.add(ldr.vel, ldr.pos[c]), [-3.94*p_ZOR, -.696*p_ZOR])]
            icrds = [np.add(np.add(ldr.vel, ldr.pos[c]), [2.29*p_ZOR, 3.27*p_ZOR]),
                     np.add(np.add(ldr.vel, ldr.pos[c]), [-3.27*p_ZOR, 2.29*p_ZOR]),
                     np.add(np.add(ldr.vel, ldr.pos[c]), [-2.29*p_ZOR, -3.27*p_ZOR]),
                     np.add(np.add(ldr.vel, ldr.pos[c]), [3.28*p_ZOR, -2.29*p_ZOR])]
        elif shape == 'platoon':
            crds = [np.add(np.add(ldr.vel, ldr.pos[c]), [4 * p_ZOR, 0]),
                    np.add(np.add(ldr.vel, ldr.pos[c]), [-4 * p_ZOR, 0]),
                    np.add(np.add(ldr.vel, ldr.pos[c]), [0, 2 * p_ZOR]),
                    np.add(np.add(ldr.vel, ldr.pos[c]), [0, -2 * p_ZOR])]
            icrds = [np.add(np.add(ldr.vel, ldr.pos[c]), [3.8 * p_ZOR, 2.2 * p_ZOR]),
                     np.add(np.add(ldr.vel, ldr.pos[c]), [4.2 * p_ZOR, -1.8 * p_ZOR]),
                     np.add(np.add(ldr.vel, ldr.pos[c]), [-3.8 * p_ZOR, 1.8 * p_ZOR]),
                     np.add(np.add(ldr.vel, ldr.pos[c]), [-4.2 * p_ZOR, -2.2 * p_ZOR])]
        else:  # triangle
            crds = [np.add(np.add(ldr.vel, ldr.pos[c]), [0, 1.73*p_ZOR_tri]),  #
                    np.add(np.add(ldr.vel, ldr.pos[c]), [-.5*p_ZOR_tri, p_ZOR_tri]),  #
                    np.add(np.add(ldr.vel, ldr.pos[c]), [.5 * p_ZOR_tri, 1.2 * p_ZOR_tri])]  #
            icrds = [np.add(np.add(ldr.vel, ldr.pos[c]), [1.5*p_ZOR_tri, -.7*p_ZOR_tri]),
                     np.add(np.add(ldr.vel, ldr.pos[c]), [-1.5*p_ZOR_tri, -.9*p_ZOR_tri]),
                     np.add(np.add(ldr.vel, ldr.pos[c]), [p_ZOR_tri, -0.75*p_ZOR_tri]),  #
                     np.add(np.add(ldr.vel, ldr.pos[c]), [.75*p_ZOR_tri, .375*p_ZOR_tri]),  #
                     np.add(np.add(ldr.vel, ldr.pos[c]), [-p_ZOR_tri, -0.225*p_ZOR_tri]),
                     np.add(np.add(ldr.vel, ldr.pos[c]), [-.1*p_ZOR_tri, -.9*p_ZOR_tri])]
        self.destination = None  # 6
        for spot in crds:  # 10
            if self.get_distance(c, spot[0], spot[1]) >= .01*p_ZOR and not self.check_spot(c, spot):
                self.destination = spot
        if self.destination is None:
            for spot in icrds:
                if self.get_distance(c, spot[0], spot[1]) >= .01*p_ZOR and not self.check_spot(c, spot):
                    self.destination = spot

    def check_leader(self, q):
        self.calc_inc(3 + len(self.v_nbrs_all))
        if q % 100 == 0:
            self.best_leader = self.leadership  # reset every so often in case the leader gets taken out
        leader = self
        for agent in self.v_nbrs_all:
            if agent.leadership >= self.best_leader:
                self.best_leader = agent.leadership
                leader = agent
        if leader == self:
            self.rank = 1
        return leader

    def check_next_up(self, q):
        leader = self
        self.calc_inc(5 + len(self.my_column_nbrs))
        for agent in self.my_column_nbrs:
            if agent.leadership >= leader.leadership:
                leader = agent
        if leader == self and not self.need_resume and len(self.my_column_nbrs) > 1:
            self.column_leader = True
        else:
            for agent in self.my_column_nbrs:
                if agent.leadership > self.leadership and not agent.column_leader:
                    leader = agent
        if self.column_leader:
            for agent in self.v_nbrs_all:
                if agent.column_leader and agent.column == 2:
                    leader = agent
        return leader

    def check_rank(self, q):
        self.calc_inc(len(self.v_nbrs_all)*2 + 5)
        for agent in self.v_nbrs_all:
            if agent.leadership > self.leadership:
                self.flanking = agent.flanking
                self.north_south = agent.north_south
        if p_s_space == 30:
            if q % 300 == 0:
                if self.rank == 1 and self.flanking % 2 == 1 and not self.avoid_obstacle and not self.need_charge and not self.need_resume:
                    self.leadership -= 1000
        else:
            if q % 3000 == 0:
                if self.rank == 1 and self.flanking % 2 == 1 and not self.avoid_obstacle and not self.need_charge and not self.need_resume:
                    self.leadership -= 1000
        self.rank = 1
        for n in self.v_nbrs_all:
            if n.leadership > self.leadership:
                self.rank += 1
        if self.rank == 1 and self.flanking % 2 == 1:
            self.pivot_point = self.est_pos[1]
        parent_ranks = [1, 1, 2, 3, 4, 1, 6, 7, 8]
        self.parent_rank = parent_ranks[self.rank - 1]
        if self.rank < 6:
            self.left_wing = True
        else:
            self.left_wing = False

    def follow(self, ldr, c):
        self.calc_inc(4)
        s_ldr = ldr
        if c == 4:
            if not self.column_leader:
                self.vel = np.divide(np.subtract(ldr.pos[c - 4], self.pos[c]), 30)
            else:
                self.calc_inc(len(self.v_nbrs_all))
                for agent in self.v_nbrs_all:
                    if agent.column_leader and agent.column == 2:
                        s_ldr = agent  # supreme leader will be yellow, leading the middle column
                if self.column == 1:
                    self.vel = np.subtract(np.add(s_ldr.pos[c], [1*math.sin(s_ldr.heading),
                                                                 -1*math.cos(s_ldr.heading)]), self.pos[c])
                else:
                    self.vel = np.subtract(np.add(s_ldr.pos[c], [-1*math.sin(s_ldr.heading),
                                                                 1*math.cos(s_ldr.heading)]), self.pos[c])

    def check_spot(self, c_pose, spot):
        self.calc_inc(4 + len(self.v_nbrs_all))
        x = spot[0]
        y = spot[1]
        occupied = False
        for n in self.v_nbrs_all:
            if not occupied:
                if x-0.5 < n.pos[c_pose][0] < x+0.5 and y-0.5 < n.pos[c_pose][1] < y+0.5:
                    occupied = True
        if occupied:
            return True
        else:
            return False

    def get_intersections(self, c, nnl, nnr, radius):
        self.calc_inc(3)
        poses = [nnl.pos[c][0], nnl.pos[c][1], nnr.pos[c][0], nnr.pos[c][1]]
        d = math.sqrt((poses[2] - poses[0]) ** 2 + (poses[3] - poses[1]) ** 2)
        if d > 2 * radius or d == 0:  # non intersecting
            return None
        else:
            self.calc_inc(4)
            h = math.sqrt(radius ** 2 - ((d ** 2) / (2 * d)) ** 2)
            return (poses[0] + ((d ** 2) / (2 * d)) * (poses[2] - poses[0]) / d) + h * (poses[3] - poses[1]) / d, \
                   (poses[1] + ((d ** 2) / (2 * d)) * (poses[3] - poses[1]) / d) - h * (poses[2] - poses[0]) / d, \
                   (poses[0] + ((d ** 2) / (2 * d)) * (poses[2] - poses[0]) / d) - h * (poses[3] - poses[1]) / d, \
                   (poses[1] + ((d ** 2) / (2 * d)) * (poses[3] - poses[1]) / d) + h * (poses[2] - poses[0]) / d

    def north_end_rule(self, c, c_axis):
        self.calc_inc(5)
        if self.NNR.avoid_obstacle:
            self.flank(v=1, d=self.flanking)
        else:
            t1 = [self.NNR.pos[c][0] + p_ZOR * math.sin(c_axis), self.NNR.pos[c][1] + p_ZOR * math.cos(c_axis)]
            t2 = [self.NNR.pos[c][0] - p_ZOR * math.sin(c_axis), self.NNR.pos[c][1] - p_ZOR * math.cos(c_axis)]
            distance_to_point1 = math.sqrt((t1[0] - self.est_pos[0]) ** 2 + (t1[1] - self.est_pos[1]) ** 2)
            distance_to_point2 = math.sqrt((t2[0] - self.est_pos[0]) ** 2 + (t2[1] - self.est_pos[1]) ** 2)
            if distance_to_point1 < distance_to_point2:
                self.vel = np.dot(p_step_size, np.subtract(t1, self.est_pos))
            else:
                self.vel = np.dot(p_step_size, np.subtract(t2, self.est_pos))

    def line_avoid(self):  # flanking 1= right, 2=down, 3=left, 4=up. maneuver around an obstacle.
        self.calc_inc(4)
        avoid_angles = [[0], [75],
                        [0], [47, 53, 46, 54, 45, 55, 44, 56, 43, 57, 42, 58, 41, 59, 40], [0]]
        new_angle = 50
        for a in avoid_angles[self.flanking]:
            self.calc_inc(1)
            if self.radar[a] == 0:
                new_angle = a
        if new_angle > 50:
            self.flank(v=1, d=2)
        else:
            self.flank(v=1, d=4)

    def set_pivot_point(self, f):
        if f != 3:
            if self.north_south == 0 and self.est_pos[1] < self.pivot_point - 1 \
                    and -.9 * p_s_space < self.est_pos[1] < .9 * p_s_space:  # prevent repeated passes of same area
                self.pivot_point += 5
            elif self.north_south == 1 and self.est_pos[1] > self.pivot_point + 1 \
                    and -.9 * p_s_space < self.est_pos[1] < .9 * p_s_space:
                self.pivot_point -= 5
            else:
                self.pivot_point = self.est_pos[1]
        else:
            if self.north_end and self.north_south == 0 and self.est_pos[1] < self.pivot_point - 1 \
                    and -.9 * p_s_space < self.est_pos[1] < .9 * p_s_space:  # prevent repeated passes of same area
                self.pivot_point += abs(self.pivot_point - self.est_pos[1])
            elif self.south_end and self.north_south == 1 and self.est_pos[1] > self.pivot_point + 1 \
                    and -.9 * p_s_space < self.est_pos[1] < .9 * p_s_space:
                self.pivot_point -= abs(self.pivot_point - self.est_pos[1])
            else:
                self.pivot_point = self.est_pos[1]

    def line_initiate_pivot(self, c):
        self.calc_inc(3)
        up_down_flanking = [4, 2]
        if self.flanking % 2 == 1 and abs(self.est_pos[0]) > 5+p_s_space:
            self.calc_inc(3)
            self.flanking = up_down_flanking[self.north_south]
            self.set_pivot_point(3)
        elif self.flanking == 2:
            if self.est_pos[1] < self.pivot_point - self.line_width or self.est_pos[1] < -1.1*p_s_space:
                self.calc_inc(3)
                if self.est_pos[0] < 0:
                    self.orig_y = self.est_pos[1]
                    self.flanking = 1
                    self.heading = 0
                else:
                    self.orig_y = self.est_pos[1]
                    self.flanking = 3
                    self.heading = np.pi
        elif self.flanking == 4:
            if self.est_pos[1] > self.pivot_point + self.line_width or self.est_pos[1] > 1.1*p_s_space:
                self.calc_inc(3)
                if self.est_pos[0] < 0:
                    self.flanking = 1
                    self.heading = 0
                else:
                    self.flanking = 3
                    self.heading = np.pi

    def line_match_neighbor(self, c):
        self.calc_inc(4)
        if self.flanking == 1 and self.north_south == 0 and self.est_pos[0] > 0:  # going right and ready to go up
            if self.NNL.flanking == 4 or self.NNR.flanking == 4:
                self.calc_inc(3)
                self.flanking = 4
                self.set_pivot_point(3)
        elif self.flanking == 1 and self.north_south == 1 and self.est_pos[0] > 0:  # going right and ready to go down
            if self.NNL.flanking == 2 or self.NNR.flanking == 2:
                self.calc_inc(3)
                self.flanking = 2
                self.set_pivot_point(3)
        elif self.flanking == 3 and self.north_south == 0 and self.est_pos[0] < 0:  # going left and ready to go up
            if self.NNL.flanking == 4 or self.NNR.flanking == 4:
                self.calc_inc(3)
                self.flanking = 4
                self.set_pivot_point(3)
        elif self.flanking == 3 and self.north_south == 1 and self.est_pos[0] < 0:  # going left and ready to go down
            if self.NNL.flanking == 2 or self.NNR.flanking == 2:
                self.calc_inc(3)
                self.flanking = 2
                self.set_pivot_point(3)
        elif self.flanking == 2 or self.flanking == 4:  # going up or down and getting ready to go left or right
            self.calc_inc(2)
            if self.est_pos[0] < 0:
                if self.NNL.flanking == 1 or self.NNR.flanking == 1:
                    self.flanking = 1  # not this one
                    self.orig_y = self.est_pos[1]
            if self.est_pos[0] > 0:
                if self.NNL.flanking == 3 or self.NNR.flanking == 3:
                    self.flanking = 3
                    self.orig_y = self.est_pos[1]

    def line_move_along_axis(self, c, c_axis):  # move along central axis using rules
        self.calc_inc(4)
        if not self.v_nbrs_all:
            self.flank(v=1, d=self.flanking)
        if self.south_end:
            self.south_end_rule(c, c_axis)
        elif self.north_end:
            self.north_end_rule(c, c_axis)
        else:
            self.middle_rule(c, c_axis)

    def reset_north_south(self, my_n_s, nnl_n_s, nnr_n_s):
        self.calc_inc(4)
        if self.est_pos[1] > .98 * p_s_space:  # reach top
            self.north_south = 1
        elif self.est_pos[1] < -0.98 * p_s_space:  # reach bottom
            self.north_south = 0
        else:  # look at appropriate neighbor
            if my_n_s == 0:
                self.north_south = nnl_n_s
            else:
                self.north_south = nnr_n_s

    def line_rules(self, c, c_axis):
        self.calc_inc(4)
        if self.need_return:
            self.check_laser(c, 3)
        else:
            self.check_laser(c, 2)
        if self.clear_ahead(a=p_flank_radars[self.flanking], cone=1):
            self.avoid_obstacle = False
        else:
            self.avoid_obstacle = True
            self.line_avoid()
            if not self.need_return:
                self.need_return = True
                self.orig_y = self.est_pos[1]
        if self.flanking == 1 or self.flanking == 3:
            if not self.avoid_obstacle:
                self.reset_north_south(self.north_south, self.NNL.north_south, self.NNR.north_south)
                if self.flanking == 3:
                    self.heading = 0.05 * (c_axis - np.pi - self.heading)
                self.calculate_spot_in_formation(c)
                if self.near_obstacle:
                    self.flank(v=1, d=self.flanking)
                else:
                    self.line_move_along_axis(c, c_axis - np.pi*(self.flanking-1)/2)
            if self.need_return:
                self.vel[1] += (self.orig_y - self.est_pos[1])
                if abs(self.orig_y - self.est_pos[1]) < 0.01:
                    if max(self.radar) == 0:
                        self.need_return = False
        if self.flanking == 2 or self.flanking == 4:
            self.flank(v=1, d=self.flanking)
        self.line_initiate_pivot(c)
        self.line_match_neighbor(c)

    def middle_rule(self, c, c_axis):
        self.calc_inc(5)
        if self.NNL.avoid_obstacle or self.NNR.avoid_obstacle:
            self.flank(v=1, d=self.flanking)
        else:
            points = self.get_intersections(c, self.NNL, self.NNR, p_ZOR)
            if points is None:  # point toward the midpoint, plus some distance along central axis
                midpoint = [0.5 * (self.NNR.pos[c][0] + self.NNL.pos[c][0]),
                            0.5 * (self.NNR.pos[c][1] + self.NNL.pos[c][1])]
                destination = np.add(midpoint, [math.cos(c_axis), math.sin(c_axis)])
                self.vel = np.dot(p_step_size, np.subtract(destination, self.pos[c]))
            else:  # point toward the spot that is closer on an angular scale
                angle1 = abs((math.atan2(points[1] - self.est_pos[1], points[0] - self.est_pos[0])) - c_axis)
                angle2 = abs((math.atan2(points[2] - self.est_pos[1], points[3] - self.est_pos[0])) - c_axis)
                if angle1 > np.pi:
                    angle1 -= 2 * np.pi
                elif angle1 < -np.pi:
                    angle1 += 2 * np.pi
                if angle2 > np.pi:
                    angle2 -= 2 * np.pi
                elif angle2 < -np.pi:
                    angle2 += 2 * np.pi
                if angle1 < angle2:
                    self.vel = np.dot(p_step_size, np.subtract([points[0] + p_step_size * math.cos(c_axis), points[1] +
                                                                p_step_size * math.sin(c_axis)], self.est_pos))
                else:
                    self.vel = np.dot(p_step_size, np.subtract([points[2] + p_step_size * math.cos(c_axis), points[3] +
                                                                p_step_size * math.sin(c_axis)], self.est_pos))

    def south_end_rule(self, c, c_axis):
        self.calc_inc(5)
        if self.NNL.avoid_obstacle:
            self.flank(v=1, d=self.flanking)
        else:
            t1 = [self.NNL.pos[c][0] + p_ZOR * math.sin(c_axis), self.NNL.pos[c][1] + p_ZOR * math.cos(c_axis)]
            t2 = [self.NNL.pos[c][0] - p_ZOR * math.sin(c_axis), self.NNL.pos[c][1] - p_ZOR * math.cos(c_axis)]
            distance_to_point1 = math.sqrt((t1[0] - self.est_pos[0]) ** 2 + (t1[1] - self.est_pos[1]) ** 2)
            distance_to_point2 = math.sqrt((t2[0] - self.est_pos[0]) ** 2 + (t2[1] - self.est_pos[1]) ** 2)
            if distance_to_point1 < distance_to_point2:
                self.vel = np.dot(p_step_size, np.subtract(t1, self.est_pos))
            else:
                self.vel = np.dot(p_step_size, np.subtract(t2, self.est_pos))

    def swarm_motion(self, lp):
        self.calc_inc(2)
        motion = [0, 0]
        center_of_mass = [0, 0]
        for n in self.v_nbrs_close:
            self.calc_inc(1)
            if p_ZOR < n.distance < p_ZOA:  # alignment
                self.calc_inc(1)
                motion = np.add(motion, np.dot(p_swarm_desires[0], [math.cos(n.heading), math.sin(n.heading)]))
        for agent in self.v_nbrs_all:
            self.calc_inc(4)
            center_of_mass = np.divide(np.add(agent.pos[lp], center_of_mass), len(self.v_nbrs_all) + 1)
            motion = np.add(motion, np.dot(p_swarm_desires[3], np.subtract(center_of_mass, self.est_pos)))  # cohesion
            if agent.distance > p_ZOA and not agent.near_obstacle:  # attraction
                motion = np.add(motion, np.dot(p_swarm_desires[1], np.subtract(agent.pos[lp], self.est_pos)))
            if agent.distance < .8 * p_ZOR_tri and not self.near_obstacle:  # separation
                motion = np.add(motion, np.dot(p_swarm_desires[2], np.subtract(self.est_pos, agent.pos[lp])))
        return motion

    def wander(self, c):
        self.calc_inc(5)
        if self.est_pos[0] > .99 * p_s_space:  # reaching a boundary of the search space
            self.wander_direction = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        elif self.est_pos[0] < -.99 * p_s_space:
            self.wander_direction = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        elif self.est_pos[1] > .99 * p_s_space:
            self.wander_direction = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        elif self.est_pos[1] < -.99 * p_s_space:
            self.wander_direction = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        self.vel = np.dot(p_step_size, self.wander_direction)
        dead_reckoning = np.add(self.vel, self.pos[c])  # true pos
        for agent in self.v_nbrs_all:  # avoid collisions
            if self.get_distance2(dead_reckoning[0], dead_reckoning[1], agent.pos[c][0], agent.pos[c][1]) < 3*p_ZOR:
                heading = math.atan2(self.vel[1], self.vel[0])
                self.vel = [p_step_size * math.cos(heading - np.pi / 2), p_step_size * math.sin(heading - np.pi / 2)]

    # high level functions

    def move_in_line_pixel(self, q, target, w, res, memory):
        global p_central_axis
        c = min(q, 4)
        if q % 5 == 0:
            self.list_visible_agents(c, s=1)
        if q % 100 == 0 and not self.accurate:
            self.error_change()
        self.rcv_gps(c)
        for agent in self.v_nbrs_all:
            if agent.accurate:
                self.dgps(agent, c)
        if not self.accurate:
            self.dgps2(c)
        if self.calculations > p_endurance:
            if not self.need_charge:
                self.saved_progress = self.est_pos
                if max(abs(self.saved_progress[0]), abs(self.saved_progress[1])) > p_s_space + 6:
                    self.saved_progress = [0, 0]
                self.need_charge = True
            if not self.need_resume:
                self.need_resume = True
        if self.need_charge or self.need_resume:
            self.rtb_recharge(c, q, formation=3)
        else:
            # self.detect_target(c, target)
            # if self.tracking:
                # p_central_axis = math.atan2(target.pos[1]-self.pos[c][1], target.pos[0] - self.pos[c][0])
            if self.need_return:
                self.check_laser(c, 3)
            else:
                self.check_laser(c, 2)
            if self.near_obstacle:
                self.snake(c, 1, self.line_width, 1, 5)
            else:
                self.line_rules(c, p_central_axis)
            if memory or p_coverage_experiment:
                my_pixel = self.mark_pixel(q, c, res)
                if memory and not self.near_obstacle and q < 100000:
                    self.memory_adjustment(my_pixel, res, strength=1/10, c=c, snake=True)
            if q > 4 and self.near_obstacle:
                self.snake_backout(c, q, self.last_backout)
        self.normalize_velocity()
        self.vel = np.subtract(self.vel, np.dot(p_step_size, w.components))  # have the wind affect the overall motion
        self.cycle_positions(q)
        self.update_data(c, target)

    def shape_flank(self, q, target, w, shape, res, memory):
        global p_ZOR, p_ZOR_tri
        c_pose = min(q, 4)
        if q % 100 == 0 and not self.accurate:
            self.error_change()
        self.rcv_gps(c_pose)
        for agent in self.v_nbrs_all:
            if agent.accurate:
                self.dgps(agent, c_pose)
        if not self.accurate:
            self.dgps2(c_pose)
        if self.calculations > p_endurance:
            if not self.need_charge:
                self.saved_progress = self.est_pos
                self.need_charge = True
            if not self.need_resume:
                self.need_resume = True
        if self.calculations > .5*p_endurance and self.est_pos[0] < -.99*p_s_space:
            if not self.need_charge:
                self.saved_progress = self.est_pos
                self.need_charge = True
            if not self.need_resume:
                self.need_resume = True
        if self.need_charge or self.need_resume:  # leader and nonleader do this
            self.check_laser(c_pose, 5)
            self.rtb_recharge(c_pose, q, formation=4)
        else:
            self.list_visible_agents(c_pose, s=0)
            leader = self.check_leader(q)
            if leader != self:
                self.reset_q = 0
            else:
                self.reset_q += 1
            if leader != self:
                if leader.need_charge:
                    self.need_charge = True
                self.check_shape_spots(c_pose, leader, shape)
            if self.destination is not None:
                self.vel = (np.subtract(self.destination, self.est_pos))
            if leader == self:
                self.check_laser(c_pose, 5)
                self.detect_target(c_pose, target)
                if self.tracking:
                    self.chase(c_pose, target)
                else:
                    if shape == 'platoon':
                        self.snake_leader(c_pose, v=min(self.reset_q/50000,.1), pivot_length=2*p_ZOR)
                    elif shape == 'triangle':
                        self.snake_leader(c_pose, v=min(self.reset_q/50000,.1), pivot_length=3.5*p_ZOR)
                    else:
                        self.snake_leader(c_pose, v=min(self.reset_q/50000,.1), pivot_length=3*p_ZOR)
                    if shape == 'platoon':
                        self.snake(c_pose, v=min(self.reset_q/50000, .1), pivot_length=2*p_ZOR, strategy=1, cone_width=20)
                    elif shape == 'triangle':
                        self.snake(c_pose, v=min(self.reset_q/50000, .1), pivot_length=3.5*p_ZOR, strategy=1, cone_width=20)
                    else:
                        self.snake(c_pose, v=min(self.reset_q/50000, .1), pivot_length=3*p_ZOR, strategy=1, cone_width=20)
            self.flanking = leader.flanking
            self.pivot_point = leader.pivot_point
            self.north_south = leader.north_south
            if leader.avoid_obstacle:  # squish/compress/expand when obstacle is clear
                if shape == 'triangle':
                    p_ZOR_tri = max(p_ZOR_tri - 0.003, 1)
                else:
                    p_ZOR = max(p_ZOR - .001, .25)
            else:
                if shape == 'triangle':
                    p_ZOR_tri = min(p_ZOR_tri + .003, 3)
                else:
                    p_ZOR = min(p_ZOR + .00005, 1)
            if memory or p_coverage_experiment:
                my_pixel = self.mark_pixel(q, c_pose, res)
                if leader == self and not self.near_obstacle and q < 100000:
                    self.memory_adjustment(my_pixel, res, strength=.003, c=c_pose, snake=True)
                    print(self.color)
        self.normalize_velocity()
        self.heading_match_velocity()
        self.vel = np.subtract(self.vel, np.dot(p_step_size, w.components))  # have the wind affect the overall motion
        self.cycle_positions(q)
        self.update_data(c_pose, target)

    def shape_column(self, q, target, w, res, memory):
        global p_ZOR, p_ZOR_tri
        c_pose = min(q, 4)
        self.list_visible_agents(c_pose, s=2)
        if q % 100 == 0 and not self.accurate:
            self.error_change()
        self.rcv_gps(c_pose)
        for agent in self.v_nbrs_all:
            if agent.accurate:
                self.dgps(agent, c_pose)
        if not self.accurate:
            self.dgps2(c_pose)
        if self.calculations > p_endurance:
            if not self.need_charge:
                self.saved_progress = self.est_pos
                if max(abs(self.saved_progress[0]), abs(self.saved_progress[1])) > p_s_space + 6:
                    self.saved_progress = [0, 0]
                self.leadership -= 500
                self.column_leader = False
                self.need_charge = True
            if not self.need_resume:
                self.need_resume = True
        if self.need_charge or self.need_resume:
            self.check_laser(c_pose, 5)
            self.rtb_recharge(c_pose, q, formation=8)
        else:
            leader = self.check_next_up(q)
            self.flanking = leader.flanking
            self.north_south = leader.north_south
            if leader != self:
                self.follow(leader, c_pose)
            else:
                if self.avoid_obstacle or self.need_return:
                    self.check_laser(c_pose, 6)
                else:
                    self.check_laser(c_pose, 4)
                self.detect_target(c_pose, target)
                if self.tracking:
                    self.chase(c_pose, target)
                else:
                    self.snake_leader(c_pose, v=.1, pivot_length=1)
                    self.snake(c_pose, v=.1, pivot_length=1, strategy=2, cone_width=5)
            if memory or p_coverage_experiment:
                my_pixel = self.mark_pixel(q, c_pose, res)
                if leader == self and not self.near_obstacle and q < 100000:
                        self.memory_adjustment(my_pixel, res, strength=.001*8, c=c_pose, snake=True)
        self.normalize_velocity()
        self.heading_match_velocity()
        self.vel = np.subtract(self.vel, np.dot(p_step_size, w.components))  # have the wind affect the overall motion
        self.cycle_positions(q)
        self.update_data(c_pose, target)


    def slawnmower(self, q, target, w, res, memory):
        if q % 100 == 0 and not self.accurate:
            self.error_change()
        c = min(q, 4)  # current position index
        self.check_laser(c, 3)
        self.rcv_gps(c)
        self.snake_leader(c, v=1, pivot_length=1)
        self.snake(c, v=1, pivot_length=1, strategy=1, cone_width=1)
        if q > 4:
            self.snake_backout(c, q, self.last_backout)
        if memory or p_coverage_experiment:
            my_pixel = self.mark_pixel(q, c, res)
            if memory and not self.near_obstacle and q < 100000:
                self.memory_adjustment(my_pixel, res, strength=1*8, c=c, snake=True)
        self.normalize_velocity()
        self.vel = np.subtract(self.vel, np.dot(p_step_size, w.components))  # have the wind affect the overall motion
        self.cycle_positions(q)
        self.update_data(c, target)

    def triangle_turn(self, q, target, w, res, memory):
        c = min(q, 4)
        if q % 100 == 0 and not self.accurate:
            self.error_change()
        self.rcv_gps(c)
        for agent in self.v_nbrs_all:
            if agent.accurate:
                self.dgps(agent, c)
        if not self.accurate:
            self.dgps2(c)
        if q % 5 == 0:
            self.list_visible_agents(c, s=3)
        self.destination = None
        self.check_rank(q)
        if self.calculations > p_endurance:
            if not self.need_charge:
                # self.leadership -= 500
                self.need_charge = True
        if self.rank == 1:
            if self.need_return:
                self.check_laser(c, rng=10)
            else:
                self.check_laser(c, rng=8)
            if self.need_charge or self.need_resume:
                if not self.need_resume:
                    self.need_resume = True
                    self.saved_progress = self.est_pos
                self.rtb_recharge(c, q, formation=7)
            else:
                self.snake_leader(c, v=.01, pivot_length=3)
                self.snake(c, v=.01, pivot_length=3, strategy=3, cone_width=40)
        else:
            if self.left_wing:
                angle_shift = np.pi / 2 + 1.04
            else:
                angle_shift = -np.pi / 2 - 1.04
            for n in self.v_nbrs_all:
                if n.rank == self.parent_rank:
                    angle = n.heading + angle_shift
                    self.destination = np.add(n.pos[c], [math.cos(angle) * p_ZOR_tri, math.sin(angle) * p_ZOR_tri])
            if self.destination is not None:
                self.vel = np.subtract(self.destination, self.est_pos)
        if memory or p_coverage_experiment:
            if not self.need_charge and not self.need_resume:
                my_pixel = self.mark_pixel(q, c, res)
                if memory and not self.near_obstacle and self.rank == 1 and q < 100000:
                    self.memory_adjustment(my_pixel, res, strength=.001*8, c=c, snake=True)
        self.normalize_velocity()
        self.heading_match_velocity()  # smoothly turn heading toward where the agent is moving
        self.vel = np.subtract(self.vel, np.dot(p_step_size, w.components))  # have the wind affect the overall motion
        self.cycle_positions(q)
        self.update_data(c, target)
        if self.need_charge:
            print(self.color)

    def wettergren_pixel(self, q, target, w, res, memory):
        if q % 100 == 0 and not self.accurate:
            self.error_change()
        lag_pose = min(q, 4 - p_time_lag)
        c_pose = min(q, 4)
        self.check_laser(c_pose, 5)
        if q % 5 == 0:
            self.list_visible_agents(c_pose, s=0)
        if self.calculations > p_endurance:
            self.need_charge = True
        if self.need_charge:
            self.rtb_recharge(c_pose, q, formation=1)
        else:
            self.detect_target(c_pose, target) # not used right now - this function allows an agent to know when it is close to the target
            if self.tracking:
                self.chase(c_pose, target)
            else:
                change_or_no = np.random.uniform(0, 1)
                if self.behavior == 1: # 1 means wandering
                    if change_or_no < .1*p_wander:
                        self.change_behavior()
                else: # if not already wandering
                    if change_or_no < 0.01:
                        self.change_behavior()
                if self.behavior == 1:
                    self.wander(c_pose)
                else:
                    if len(self.v_nbrs_all) == 0:
                        self.wander(c_pose)
                    else:
                        self.vel = self.swarm_motion(lag_pose)
                        group_movement = [np.subtract([p_s_space, -p_s_space], self.est_pos),
                                          np.subtract([p_s_space, p_s_space], self.est_pos),
                                          np.subtract([-p_s_space, p_s_space], self.est_pos),
                                          np.subtract([-p_s_space, -p_s_space], self.est_pos)]
                        self.vel = np.add(self.vel, group_movement[(q//100)%4])
                if memory or p_coverage_experiment:
                    my_pixel = self.mark_pixel(q, c_pose, res)
                    if memory and not self.near_obstacle and q < 100000:
                        self.memory_adjustment(my_pixel, res, strength=1*8, c=c_pose, snake=False)
        self.stay_in_bounds()
        self.make_safe()
        self.normalize_velocity()
        self.heading_match_velocity()
        self.vel = np.subtract(self.vel, np.dot(p_step_size, w.components))  # have the wind affect the overall motion
        self.cycle_positions(q)
        self.rcv_gps(c_pose)
        for agent in self.v_nbrs_close:
            if agent.accurate:
                self.dgps(agent, c_pose)
        if not self.accurate:
            self.dgps2(c_pose)
        self.update_data(c_pose, target)

def demo(t, target, resolution, rep, function, memory):
    global p_wander
    if memory == 1:
        mem = True
    else:
        mem = False
    if p_wind:
        magnitude, direction = get_xy('target.csv')
        wind.direction = float(direction)
        wind.force = 0.0003 * float(magnitude)  # 30 kts of wind = 1 kt of current
        wind.components = [wind.force * math.cos(wind.direction), wind.force * math.sin(wind.direction)]
    if p_coverage_experiment:
        g_data[rep, t] = (float(g_pixels_explored / (resolution ** 2)))
    if function == 1:  # baseline or baseline pixel
        p_wander = 0
        for agent in g_agents:
            agent.behavior = 1
    else:
        p_wander = .1
    for agent in g_agents:
        if function == 1:
            agent.wettergren_pixel(t, target, wind, resolution, mem)
        elif function == 2:
            agent.wettergren_pixel(t, target, wind, resolution, mem)
        elif function == 3:
            agent.move_in_line_pixel(t, target, wind, resolution, mem)
        elif function == 4:
            agent.shape_flank(t, target, wind, 'circle', resolution, mem)
        elif function == 5:
            agent.shape_flank(t, target, wind, 'triangle', resolution, mem)
        elif function == 6:
            agent.shape_flank(t, target, wind, 'platoon', resolution, mem)
        elif function == 7:
            agent.triangle_turn(t, target, wind, resolution, mem)
        elif function == 8:
            agent.shape_column(t, target, wind, resolution, mem)
        elif function == 9:
            agent.slawnmower(t, target, wind, resolution, mem)
    for target in g_hidden_objects:
        target.detect_agents(t)
        #target.move_around(t)
        target.stay_in_bounds()
        target.normalize_vel()
        target.update_pose()
    if not p_coverage_experiment and t % 20 == 0 and memory == True:  # look into this - why do they have to freshen up if it is not a memory experiment?
        for pixel in g_pixels:
            pixel.freshen_up(t)
    if p_plot and t % 10 == 0:
        make_plot(t, function, mem)


def make_plot(t, function, mem):
    agent_marker_sizes = [4, 4, 4, 4, 4, 4, 4, 4, 20]
    current = min(t, 4)
    if t == 0:
        pixel_color = '1.0'
        for pixel in g_pixels:
            pixel.pheromones = 0
            plt.plot(pixel.centerx, pixel.centery, marker='s', markeredgecolor=pixel_color,
                     markerfacecolor=pixel_color, markersize=10)
    for pixel in g_pixels:
        if pixel.pheromones > 1:
            pixel_color = '0.9'
            if pixel.pheromones > 5:
                pixel_color = '0.8'
                if pixel.pheromones > 10:
                    pixel_color = '0.7'
            plt.plot(pixel.centerx, pixel.centery, marker='s', markeredgecolor=pixel_color,
                     markerfacecolor=pixel_color, markersize=20)
    for targ in g_hidden_objects:
        plt.plot(targ.pos[0], targ.pos[1], color='red', marker='o', markerfacecolor='r', markersize=21, clip_on=False)
        plt.plot(targ.pos[0], targ.pos[1], color='white', marker='o', markerfacecolor='w', markersize=14, clip_on=False)
        plt.plot(targ.pos[0], targ.pos[1], color='red', marker='o', markerfacecolor='r', markersize=7, clip_on=False)
    if t > 9 and t % 200 == 0 and p_show_history:
        for j in range(max(t - 1500, int(t / 3)), t):
            for agent in g_agents:  # past positions
                plt.xlim([-1 * p_s_space, p_s_space])
                plt.ylim([-1 * p_s_space, p_s_space])
                plt.plot(agent.pos[j][0], agent.pos[j][1], marker='o', markeredgecolor='0.8',
                         markerfacecolor='0.8', markersize=4, clip_on=False)
    for agent in g_agents:
        plt.xlim([-1 * p_s_space, p_s_space])
        plt.ylim([-1 * p_s_space, p_s_space])
        plt.plot(agent.pos[current][0], agent.pos[current][1], marker='o', markeredgecolor=agent.color,
                 markerfacecolor=agent.color, markersize=agent_marker_sizes[function - 1], clip_on=False)
        if p_gps_experiment:
            plt.plot(agent.est_pos[0], agent.est_pos[1], marker='o', markeredgecolor=agent.color, markerfacecolor='None',
                     markersize=agent_marker_sizes[function-1], clip_on=False)
            plt.plot([agent.pos[current][0], agent.est_pos[0]], [agent.pos[current][1], agent.est_pos[1]],
                     color=agent.color, linestyle='dashed', clip_on=False)
    for obs in g_obstacles:
        plt.xlim([-1 * p_s_space, p_s_space])
        plt.ylim([-1 * p_s_space, p_s_space])
        plt.plot(obs.pos[0], obs.pos[1], marker='o', markeredgecolor='black', markerfacecolor='.5',
                 markersize=o_m_size, clip_on=False)
    plt.grid()
    if mem:
        plt.text(0, p_s_space, p_labels_mem[function - 1], ha='center', va='bottom')
    else:
        plt.text(0, p_s_space, p_labels_no_mem[function - 1], ha='center', va='bottom')
    plt.pause(0.01)
    plt.clf()


def reset_data():
    global g_percent_distance, g_data, g_agent_distances, g_agents, g_hidden_objects, g_obstacles, g_pixels
    g_agent_distances.clear()
    g_agents.clear()
    g_pixels.clear()
    g_obstacles.clear()
    g_hidden_objects.clear()
    g_percent_distance = 100


def data(func, n_ag, n_obs, memory, resolution, rep):
    global g_data, g_found_target
    reset_data()
    for o in np.arange(n_obs):
        obstacle = Obstacle()
    if func == 8:  # special colors for platoon column
        columns = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    colors = ['g', 'b', 'y', 'c', 'r', 'k', '0.2', '0.4', '0.6']
    for k in np.arange(n_ag):
        particle = Agent()
        particle.color = colors[k]   # change this to keep color static
        if func == 8:
            particle.column = columns[k]
        if len(g_agents) == 1:
        #if len(g_agents) >= 0:
            particle.accurate = True
            particle.gps_error = [0, 0]
        particle.name = str(k)
    for b in np.arange(resolution):
        for p in np.arange(resolution):
            name = Pixel(b, p, resolution)
            name.pheromones = 0
    target3 = Hidden_Object(np.random.uniform(-.75, .9)*p_s_space, np.random.uniform(-.8, .8)*p_s_space)
    #target3 = Hidden_Object(25, -20)
    iterations = np.arange(p_iterations)
    g_found_target = False
    for h in iterations:
        if not g_found_target:
            if h == p_iterations - 1:
                print('target not found')
            demo(h, target3, resolution, rep, func, memory)
            if g_percent_distance < p_detection_range[func] and not p_coverage_experiment:
                g_data[rep, 0] = h
                g_found_target = True
                print(h)
            if len(g_agent_distances) >= 1 and p_distance_experiment:  # if you want to see step by step best distances
                best_agent_distance = min(g_agent_distances)
                g_data[rep, h + 1] = best_agent_distance
        else:
            break


def run(z, func_number, n, n_obstacles, memory):
    global g_pixels_explored
    global p_iterations
    '''if func_number >= 7:
        p_iterations = 250000
    else:
        p_iterations = 150000'''
    if p_coverage_experiment:
        for pixel in g_pixels:
            pixel.explored = False
    g_pixels_explored = 0
    print('rep: ' + str(z) + ' function: ' + str(func_number) + ' size: ' + str(p_s_space) + ' obstacles: ' +
          str(n_obstacles) + ' memory: ' + str(memory) + '  ' + str(time.time() - start))
    data(func_number, n, n_obstacles, memory, 1, z)


# _thread.start_new_thread(start_drag, ())
wind = Wind()
f_numbers = [1,2,3, 4, 5, 6,7,8, 1, 3, 4,5,7,8,1,7,6,1,3,7,5,4,7,4,1,4,7,4,7,7,7,2,2,1,2,3,2,1,4,1,7,3,8,7,4,8,7,7,1,3,6,4,2,8,7,2,1,8,3,1,8,8,4,5,7,6,7,4,3,3,2,6,7,2,5,1,1,6,3,5,8,5,8,6,3,3,6,4,6,1,4,1,2,8,6,2,1,3,4,4,8,8,4,1,5,8,4,7,5,3,1,7,2,3,4,5,6,3,3,2,8,6,5,3,6,5,5,3,2,2,7,3,4,4,6,7,2,1,2,7,4,1,3,6,7,3,6,3,1,8,5,6,1,7,5,5,3,1,6,2,1,6,7,1,2,8,6,3,5,7,5,1,7,3,7,4,8,6,4,4,7,5,3,2,5,3,2,1,4,5,2,3,4,6,6,4,1,4,8,5,3,7,4,2,4,2,8,6,4,6,5,1,4,8,1,8,1,6,3,7,3,1,5,6,6,5,8,5,7,1,3,2,4,4,7,6,4,6,2,6,3,7,1,5,2,4,8,5,7,2,1,2,8,4,2,8,6,4,5,3,3,6,4,1,4,8,3,2,5,4,2,1,6,6,8,5,8,6,7,4,5,7,3,2,6,8,2,5,2,1,2,7,5,1,6,3,3,1,5,3,2,3,6,2,4,4,7,1,4,3,6,8,7,4,5,7,5,8,2,5,1,6,7,3,6,6,3,3,2,7,2,6,5,2,8,1,7,6,8,8,6,6,5,2,1,4,1,1,8,5,5,7,3,5,5,5,7,6,8,3,4,3,2,6,6,5,6,1,7,4,7,7,8,1,7,8,8,3,4,6,8,2,3,2,5,8,8,1,2,3,6,5,3,1,2,4,2,6,8,8,5,8,2,1,6,6,8,2,2,1,3,5,6,4,6,6,4,2,6,1,4,8,7,4,2,6,1,6,3,8,5,5,4,6,7,2,6,2,5,2,1,6,7,4,5,6,8,5,6,7,7,4,5,1,2,6,4,7,1,5,5,7,8,6,4,4,8,2,4,1,4,4,8,7,2,2,3,2,6,6,2,5,5,5,5,1,4,2,5,3,2,2,6,5,7,2,5,4,3,3,1,6,3,4,7,8,2,6,1,3,4,3,6,7,8,2,6,6,7,5,2,2,4,7,1,6,3,8,2,5,2,8,1,7,4,5,8,7,5,3,1,2,3,5,3,5,6,7,4,7,3,4,5,5,5,1,3,8,7,3,8,3,6,7,3,7,8,7,1,1,5,6,1,6,8,4,8,2,5,1,8,2,5,4,8,3,5,8,5,7,8,8,7,1,7,6,8,7,6,8,8,1,3,5,2,2,7,1,3,2,6,7,7,7,2,1,5,3,4,8,3,6,4,6,3,3,6,2,1,8,7,6,8,7,4,7,8,5,4,3,6,8,3,7,2,2,7,8,2,7,7,2,6,3,2,2,4,7,1,3,1,7,5,5,6,7,5,5,1,5,5,6,8,7,6,7,2,4,6,6,6,6,6,5,7,2,2,3,8,1,7,1,5,5,5,1,2,1,1,5,8,2,5,4,4,4,7,7,7,4,1,5,5,3,3,7,2,3,6,8,5,1,7,7,6,1,2,8,2,3,4,2,3,8,5,3,8,1,8,7,1,3,2,3,3,2,8,4,4,7,8,1,5,1,6,8,2,2,2,7,7,6,1,2,7,1,8,1,1,8,1,1,6,2,3,6,2,8,4,1,4,2,5,2,6,7,7,1,1,2,2,8,4,7,5,5,6,5,6,4,8,1,8,2,2,2,7,6,7,4,7,8,5,5,6,2,5,1,3,3,3,5,4,3,1,6,3,2,2,1,8,7,8,1,3,5,6,4,8,6,8,2,7,3,5,8,2,6,6,8,6,3,6,8,2,1,2,8,2,1,1,4,6,1,8,4,1,5,8,7,8,1,4,5,3,8,4,5,2,6,6,4,4,1,3,3,8,5,8,7,7,4,6,1,8,5,7,7,3,4,5,3,4,6,3,7,3,2,8,3,7,5,6,1,8,3,1,1,1,5,6,8,6,3,1,7,3,2,5,4,2,1,5,6,6,4,2,7,5,2,8,8,2,3,1,8,8,3,3,3,3,2,8,1,5,2,6,8,2,6,8,5,1,8,3,5,6,3,1,3,6,5,5,6,8,3,4,1,2,5,5,5,2,1,5,5,4,5,8,8,7,2,2,1,2,5,1,5,1,7,3,8,6,6,3,8,7,8,7,4,6,3,1,3,3,2,6,4,1,2,6,7,7,7,6,2,3,8,8,3,2,2,8,3,1,5,7,8,5,5,6,8,8,3,4,7,2,4,7,3,7,2,2,6,4,3,4,7,2,6,8,2,4,5,8,1,7,1,4,2,8,3,7,8,3,7,4,4,3,5,8,1,2,6,8,8,7,7,7,6,8,4,4,3,2,4,4,2,2,1,6,4,4,5,1,3,2,7,1,1,1,6,4,2,3,8,3,1,2,6,4,5,1,8,2,2,5,6,4,3,5,4,3,8,3,7,1,4,8,1,4,5,7,2,3,7,8,4,1,4,4,2,5,6,6,6,1,3,1,6,4,1,2,5,2,7,2,5,4,4,8,1,3,7,6,2,1,1,2,8,3,6,7,8,5,5,8,5,1,4,3,8,2,5,5,1,7,7,4,4,5,1,5,4,5,4,7,8,4,1,3,6,7,8,8,6,6,3,2,3,2,1,1,4,3,6,5,8,6,7,5,1,1,8,7,1,5,3,5,3,4,2,7,7,5,4,7,4,5,3,6,7,5,1,5,1,6,1,5,8,1,2,3,4,3,2,6,7,7,5,6,8,3,3,3,3,6,5,4,1,4,4,4,1,7,4,8,8,3,1,2,5,2,5,8,6,3,1,1,1,7,6,7,1,4,8,4,7,7,7,4,1,5,1,3,2,7,2,2,4,6,2,1,7,7,6,1,7,3,2,7,6,8,5,1,8,8,6,3,2,1,5,4,8,4,8,4,4,3,2,5,4,6,6,7,8,5,8,7,4,4,8,8,2,4,1,2,6,5,8,6,6,1,1,2,8,3,8,8,4,2,6,1,8,1,7,8,3,7,1,4,5,1,7,3,8,2,4,2,4,6,4,6,4,1,4,3,8,4,2,3,1,5,6,2,4,2,8,8,5,2,6,5,1,4,2,2,8,4,5,2,7,8,1,5,7,5,4,1,6,8,3,6,1,6,7,3,5,7,3,3,7,5,8,2,2,4,7,7,2,1,6,7,5,8,2,8,6,7,7,5,3,5,6,8,2,6,6,2,5,1,7,2,5,3,6,3,4,2,6,4,7,5,1,1,2,3,8,5,8,7,3,4,3,5,6,4,2,5,6,6,2,8,5,8,6,1,2,8,2,2,8,6,5,7,1,7,3,5,4,8,1,2,2,1,2,7,2,7,3,1,6,8,3,5,8,8,6,1,4,3,2,1,6,3,3,4,1,5,2,5,4,6,4,7,3,4,3,3,4,4,2,5,8,1,8,8,5,2,2,5,1,6,1,6,6,6,2,8,5,6,3,6,6,2,7,4,3,6,2,8,4,7,1,4,1,2,7,5,7,7,6,8,4,2,4,4,1,5,5,3,8,3,7,4,5,1,6,6,7,7,7,4,2,3,6,8,3,5,7,6,6,4,7,5,2,1,5,6,5,5,8,4,3,4,2,2,7,3,8,7,8,6,1,1,8,5,4,4,7,4,4,8,3,3,7,6,2,2,6,7,5,5,7,2,6,5,3,7,5,4,4,3,3,6,1,5,1,1,7,2,4,3,6,4,8,8,5,6,7,1,4,1,7,2,8,4,5,3,8,6,3,4,4,6,1,1,8,7,3,8,1,6,5,7,7,5,7,4,5,3,1,1,2,7,4,3,4,5,3,8,5,8,2,2,8,3,1,3,2,3,6,3,7,6,4,1,8,1,8,1,1,7,2,4,4,1,3,1,2,5,4,6,5,7,5,1,8,3,3,5,3,2,2,8,2,1,2,7,5,5,4,1,3,4,1,7,7,6,3,1,3,3,1,7,5,5,1,2,2,5,6,5,2,7,5,3,2,4,6,5,4,2,8,6,1,2,3,6,4,4,5,3,6,4,1,8,3,3,7,3,8,8,2,3,2,4,6,4,2,2,5,3,7,4,6,2,2,1,5,5,4,8,8,7,5,4,8,1,7,4,4,1,5,6,3,2,5,4,8,7,5,3,4,4,3,7,5,8,5,6,5,1,1,4,6,3,1,7,2,2,6,7,1,3,2,5,6,2,4,7,2,4,3,4,3,1,3,4,4,5,5,2,4,7,8,4,1,1,7,1,8,7,8,5,1,6,5,5,3,3,1,3,7,7,7,7,4,1,8,4,3,4,7,1,4,5,7,1,8,5,6,1,6,2,4,5,8,5,6,4,2,8,7,1,3,5,4,6,2,2,6,6,3,8,2,7,4,3,6,4,5,2,7,8,7,6,8,8,3,4,5,3,5,5,6,7,8,2,1,7,5,7,1,7,2,8,6,5,1,7,4,6,1,2,4,6,3,2,5,6,3,2,6,2,2,6,2,7,1,1,2,5,7,7,7,6,4,3,6,2,3,7,1,1,2,4,7,2,2,3,4,6,4,2,8,3,3,2,7,1,5,4,1,6,2,6,2,1,8,4,2,4,2,2,8,3,4,3,6,6,5,4,1,7,3,8,1,4,3,5,5,4,2,3,4,7,2,5,3,4,2,3,7,5,3,7,2,7,4,8,1,1,1,7,4,4,4,7,8,1,1,4,5,1,7,8,4,3,8,8,4,4,3,6,6,3,3,1,7,2,6,6,2,3,8,3,5,7,6,3,7,1,3,1,5,6,6,1,6,5,3,8,2,5,1,2,8,2,5,3,8,6,5,4,5,2,2,1,8,1,6,6,2,4,8,2,8,6,8,6,3,7,5,1,1,8,7,6,6,2,5,4,4,5,6,7,4,6,4,8,4,5,1,2,2,6,2,2,4,5,5,7,7,7,7,5,5,4,6,1,4,6,4,1,7,3,3,3,3,8,1,6,1,5,4,5,2,8,1,6,5,1,8,8,2,6,2,2,5,8,3,3,5,6,4,8,3,6,1,8,8,5,3,2,8,5,6,8,2,7,7,4,1,6,7,1,6,1,5,3,3,2,7,1,1,4,8,7,1,1,7,7,3,2,3,6,6,5,1,5,6,7,1,4,7,3,6,6,3,8,8,6,8,4,5,3,8,7,1,5,8,6,4,8,2,8,5,3,5,3,1,8,6,4,4,8,3,4,3,3,1,8,4,8,5,3,5,8,6,6,4,4,4,7,1,3,8,6,5,1,5,3,6,6,4,7,1,2,1,7,5,6,5,3,1,1,4,2,2,3,3,1,3,2,1,1,4,4,4,8,1,5,7,2,6,8,8,8,4,4,2,6,2,1,6,7,3,5,3,7,2,3,6,8,6,3,3,3,2,8,7,2,5,4,3,1,8,1,3,8,3,7,8,2,1,7,2,1,7,6,8,6,8,8,2,8,3,3,4,8,5,8,8,8,5,7,7,2,8,1,8,1,5,4,7,8,4,4,6,1,7,5,6,6,2,2,3,7,5,6,4,1,6,7,5,3,7,3,4,2,8,8,2,3,7,7,2,5,1,1,7,8,5,6,3,3,2,3,4,4,1,8,2,5,7,1,3,1,5,1,5,5,6,6,5,2,5,6,4,2,7,7,8,3,4,6,4,2,3,2,6,1,5,6,3,5,6,5,2,4,4,5,3,2,8,6,6,3,4,5,8,7,1,8,4,5,2,3,6,1,8,2,2,5,1,4,7,4,8,7,1,7,8,3,8,5,6,2,7,4,1,6,6,3,3,7,8,7,8,3,5,6,7,1,7,1,1,7,6,2,2,1,8,5,5,7,4,5,6,5,4,6,6,8,4,7,2,8,3,8,7,5,2,1,8,7,5,3,4,3,8,6,8,8,6,5,2,5,8,2,4,6,4,3,8,7,6,5,7,7,5,4,8,5,1,2,7,5,1,4,4,2,4,7,3,7,2,8,4,3,1,8,1,4,3,6,1,4,4,5,5,4,5,6,3,8,1,2,8,2,6,8,3,5,4,5,1,5,8,7,4,2,4,6,7,3,7,1,3,1,1,4,3,5,8,5,2,4,3,6,5,6,5,6,2,4,1,1,3,5,7,8,3,1,4,4,6,7,6,6,7,2,4,1,5,1,4,1,2,1,6,1,2,8,6,5,4,3,8,7,6,3,1,7,2,6,5,6,5,5,6,7,8,1,8,3,8,8,2,3,2,8,3,2,8,4,8,2,4,1,6,7,8,7,5,1,6,3,5,3,3,4,2,5,2,1,5,4,3,8,7,8,6,2,2,1,3,7,5,1,3,7,8,3,1,3,3,7,7,5,7,7,7,8,5,7,2,4,4,1,6,5,4,7,6,1,8,3,7,6,3,4,6,1,6,6,3,4,7,7,3,7,5,2,6,3,6,8,8,2,4,7,7,8,4,6,3,6,3,4,4,4,2,5,6,6,5,3,1,1,5,3,4,3,4,5,2,3,3,3,5,5,2,2,3,7,6,5,2,2,8,7,1,4,3,8,5,6,5,1,8,7,4,8,1,2,6,8,8,2,5,2,4,2,3,8,6,3,3,8,7,2,7,5,6,4,1,8,8,7,4,4,4,4,1,6,4,7,1,5,6,5,8,2,8,1,6,7,4,7,4,5,3,8,3,2,1,8,7,7,4,5,8,6,7,6,6,1,8,6,1,7,6,3,5,5,7,4,4,8,3,7,2,2,3,6,6,3,1,1,2,1,7,1,3,2,3,2,1,7,7,7,8,6,7,3,7,7,7,6,6,1,2,4,8,4,4,7,3,3,5,8,3,2,5,3,6,8,6,1,4,4,1,7,1,4,4,8,1,3,8,8,7,6,2,2,8,4,1,8,1,6,3,8,5,2,5,4,3,1,1,6,6,2,7,5,2,4,5,6,1,7,1,4,1,1,5,6,8,4,4,5,5,3,3,8,8,7,7,8,8,3,7,3,1,8,4,6,7,8,2,2,2,3,6,2,5,2,7,8,8,6,1,7,8,8,3,2,5,1,3,1,2,7,7,4,7,8,6,7,7,4,2,7,4,1,5,8,7,4,1]
#f_numbers = [6,5,8,3,5,3,7,3,4,5,7,8,1,7,6,1,3,7,5,4,7,4,1,4,7,4,7,7,7,2,2,1,2,3,2,1,4,1,7,3,8,7,4,8,7,7,1,3,6,4,2,8,7,2,1,8,3,1,8,8,4,5,7,6,7,4,3,3,2,6,7,2,5,1,1,6,3,5,8,5,8,6,3,3,6,4,6,1,4,1,2,8,6,2,1,3,4,4,8,8,4,1,5,8,4,7,5,3,1,7,2,3,4,5,6,3,3,2,8,6,5,3,6,5,5,3,2,2,7,3,4,4,6,7,2,1,2,7,4,1,3,6,7,3,6,3,1,8,5,6,1,7,5,5,3,1,6,2,1,6,7,1,2,8,6,3,5,7,5,1,7,3,7,4,8,6,4,4,7,5,3,2,5,3,2,1,4,5,2,3,4,6,6,4,1,4,8,5,3,7,4,2,4,2,8,6,4,6,5,1,4,8,1,8,1,6,3,7,3,1,5,6,6,5,8,5,7,1,3,2,4,4,7,6,4,6,2,6,3,7,1,5,2,4,8,5,7,2,1,2,8,4,2,8,6,4,5,3,3,6,4,1,4,8,3,2,5,4,2,1,6,6,8,5,8,6,7,4,5,7,3,2,6,8,2,5,2,1,2,7,5,1,6,3,3,1,5,3,2,3,6,2,4,4,7,1,4,3,6,8,7,4,5,7,5,8,2,5,1,6,7,3,6,6,3,3,2,7,2,6,5,2,8,1,7,6,8,8,6,6,5,2,1,4,1,1,8,5,5,7,3,5,5,5,7,6,8,3,4,3,2,6,6,5,6,1,7,4,7,7,8,1,7,8,8,3,4,6,8,2,3,2,5,8,8,1,2,3,6,5,3,1,2,4,2,6,8,8,5,8,2,1,6,6,8,2,2,1,3,5,6,4,6,6,4,2,6,1,4,8,7,4,2,6,1,6,3,8,5,5,4,6,7,2,6,2,5,2,1,6,7,4,5,6,8,5,6,7,7,4,5,1,2,6,4,7,1,5,5,7,8,6,4,4,8,2,4,1,4,4,8,7,2,2,3,2,6,6,2,5,5,5,5,1,4,2,5,3,2,2,6,5,7,2,5,4,3,3,1,6,3,4,7,8,2,6,1,3,4,3,6,7,8,2,6,6,7,5,2,2,4,7,1,6,3,8,2,5,2,8,1,7,4,5,8,7,5,3,1,2,3,5,3,5,6,7,4,7,3,4,5,5,5,1,3,8,7,3,8,3,6,7,3,7,8,7,1,1,5,6,1,6,8,4,8,2,5,1,8,2,5,4,8,3,5,8,5,7,8,8,7,1,7,6,8,7,6,8,8,1,3,5,2,2,7,1,3,2,6,7,7,7,2,1,5,3,4,8,3,6,4,6,3,3,6,2,1,8,7,6,8,7,4,7,8,5,4,3,6,8,3,7,2,2,7,8,2,7,7,2,6,3,2,2,4,7,1,3,1,7,5,5,6,7,5,5,1,5,5,6,8,7,6,7,2,4,6,6,6,6,6,5,7,2,2,3,8,1,7,1,5,5,5,1,2,1,1,5,8,2,5,4,4,4,7,7,7,4,1,5,5,3,3,7,2,3,6,8,5,1,7,7,6,1,2,8,2,3,4,2,3,8,5,3,8,1,8,7,1,3,2,3,3,2,8,4,4,7,8,1,5,1,6,8,2,2,2,7,7,6,1,2,7,1,8,1,1,8,1,1,6,2,3,6,2,8,4,1,4,2,5,2,6,7,7,1,1,2,2,8,4,7,5,5,6,5,6,4,8,1,8,2,2,2,7,6,7,4,7,8,5,5,6,2,5,1,3,3,3,5,4,3,1,6,3,2,2,1,8,7,8,1,3,5,6,4,8,6,8,2,7,3,5,8,2,6,6,8,6,3,6,8,2,1,2,8,2,1,1,4,6,1,8,4,1,5,8,7,8,1,4,5,3,8,4,5,2,6,6,4,4,1,3,3,8,5,8,7,7,4,6,1,8,5,7,7,3,4,5,3,4,6,3,7,3,2,8,3,7,5,6,1,8,3,1,1,1,5,6,8,6,3,1,7,3,2,5,4,2,1,5,6,6,4,2,7,5,2,8,8,2,3,1,8,8,3,3,3,3,2,8,1,5,2,6,8,2,6,8,5,1,8,3,5,6,3,1,3,6,5,5,6,8,3,4,1,2,5,5,5,2,1,5,5,4,5,8,8,7,2,2,1,2,5,1,5,1,7,3,8,6,6,3,8,7,8,7,4,6,3,1,3,3,2,6,4,1,2,6,7,7,7,6,2,3,8,8,3,2,2,8,3,1,5,7,8,5,5,6,8,8,3,4,7,2,4,7,3,7,2,2,6,4,3,4,7,2,6,8,2,4,5,8,1,7,1,4,2,8,3,7,8,3,7,4,4,3,5,8,1,2,6,8,8,7,7,7,6,8,4,4,3,2,4,4,2,2,1,6,4,4,5,1,3,2,7,1,1,1,6,4,2,3,8,3,1,2,6,4,5,1,8,2,2,5,6,4,3,5,4,3,8,3,7,1,4,8,1,4,5,7,2,3,7,8,4,1,4,4,2,5,6,6,6,1,3,1,6,4,1,2,5,2,7,2,5,4,4,8,1,3,7,6,2,1,1,2,8,3,6,7,8,5,5,8,5,1,4,3,8,2,5,5,1,7,7,4,4,5,1,5,4,5,4,7,8,4,1,3,6,7,8,8,6,6,3,2,3,2,1,1,4,3,6,5,8,6,7,5,1,1,8,7,1,5,3,5,3,4,2,7,7,5,4,7,4,5,3,6,7,5,1,5,1,6,1,5,8,1,2,3,4,3,2,6,7,7,5,6,8,3,3,3,3,6,5,4,1,4,4,4,1,7,4,8,8,3,1,2,5,2,5,8,6,3,1,1,1,7,6,7,1,4,8,4,7,7,7,4,1,5,1,3,2,7,2,2,4,6,2,1,7,7,6,1,7,3,2,7,6,8,5,1,8,8,6,3,2,1,5,4,8,4,8,4,4,3,2,5,4,6,6,7,8,5,8,7,4,4,8,8,2,4,1,2,6,5,8,6,6,1,1,2,8,3,8,8,4,2,6,1,8,1,7,8,3,7,1,4,5,1,7,3,8,2,4,2,4,6,4,6,4,1,4,3,8,4,2,3,1,5,6,2,4,2,8,8,5,2,6,5,1,4,2,2,8,4,5,2,7,8,1,5,7,5,4,1,6,8,3,6,1,6,7,3,5,7,3,3,7,5,8,2,2,4,7,7,2,1,6,7,5,8,2,8,6,7,7,5,3,5,6,8,2,6,6,2,5,1,7,2,5,3,6,3,4,2,6,4,7,5,1,1,2,3,8,5,8,7,3,4,3,5,6,4,2,5,6,6,2,8,5,8,6,1,2,8,2,2,8,6,5,7,1,7,3,5,4,8,1,2,2,1,2,7,2,7,3,1,6,8,3,5,8,8,6,1,4,3,2,1,6,3,3,4,1,5,2,5,4,6,4,7,3,4,3,3,4,4,2,5,8,1,8,8,5,2,2,5,1,6,1,6,6,6,2,8,5,6,3,6,6,2,7,4,3,6,2,8,4,7,1,4,1,2,7,5,7,7,6,8,4,2,4,4,1,5,5,3,8,3,7,4,5,1,6,6,7,7,7,4,2,3,6,8,3,5,7,6,6,4,7,5,2,1,5,6,5,5,8,4,3,4,2,2,7,3,8,7,8,6,1,1,8,5,4,4,7,4,4,8,3,3,7,6,2,2,6,7,5,5,7,2,6,5,3,7,5,4,4,3,3,6,1,5,1,1,7,2,4,3,6,4,8,8,5,6,7,1,4,1,7,2,8,4,5,3,8,6,3,4,4,6,1,1,8,7,3,8,1,6,5,7,7,5,7,4,5,3,1,1,2,7,4,3,4,5,3,8,5,8,2,2,8,3,1,3,2,3,6,3,7,6,4,1,8,1,8,1,1,7,2,4,4,1,3,1,2,5,4,6,5,7,5,1,8,3,3,5,3,2,2,8,2,1,2,7,5,5,4,1,3,4,1,7,7,6,3,1,3,3,1,7,5,5,1,2,2,5,6,5,2,7,5,3,2,4,6,5,4,2,8,6,1,2,3,6,4,4,5,3,6,4,1,8,3,3,7,3,8,8,2,3,2,4,6,4,2,2,5,3,7,4,6,2,2,1,5,5,4,8,8,7,5,4,8,1,7,4,4,1,5,6,3,2,5,4,8,7,5,3,4,4,3,7,5,8,5,6,5,1,1,4,6,3,1,7,2,2,6,7,1,3,2,5,6,2,4,7,2,4,3,4,3,1,3,4,4,5,5,2,4,7,8,4,1,1,7,1,8,7,8,5,1,6,5,5,3,3,1,3,7,7,7,7,4,1,8,4,3,4,7,1,4,5,7,1,8,5,6,1,6,2,4,5,8,5,6,4,2,8,7,1,3,5,4,6,2,2,6,6,3,8,2,7,4,3,6,4,5,2,7,8,7,6,8,8,3,4,5,3,5,5,6,7,8,2,1,7,5,7,1,7,2,8,6,5,1,7,4,6,1,2,4,6,3,2,5,6,3,2,6,2,2,6,2,7,1,1,2,5,7,7,7,6,4,3,6,2,3,7,1,1,2,4,7,2,2,3,4,6,4,2,8,3,3,2,7,1,5,4,1,6,2,6,2,1,8,4,2,4,2,2,8,3,4,3,6,6,5,4,1,7,3,8,1,4,3,5,5,4,2,3,4,7,2,5,3,4,2,3,7,5,3,7,2,7,4,8,1,1,1,7,4,4,4,7,8,1,1,4,5,1,7,8,4,3,8,8,4,4,3,6,6,3,3,1,7,2,6,6,2,3,8,3,5,7,6,3,7,1,3,1,5,6,6,1,6,5,3,8,2,5,1,2,8,2,5,3,8,6,5,4,5,2,2,1,8,1,6,6,2,4,8,2,8,6,8,6,3,7,5,1,1,8,7,6,6,2,5,4,4,5,6,7,4,6,4,8,4,5,1,2,2,6,2,2,4,5,5,7,7,7,7,5,5,4,6,1,4,6,4,1,7,3,3,3,3,8,1,6,1,5,4,5,2,8,1,6,5,1,8,8,2,6,2,2,5,8,3,3,5,6,4,8,3,6,1,8,8,5,3,2,8,5,6,8,2,7,7,4,1,6,7,1,6,1,5,3,3,2,7,1,1,4,8,7,1,1,7,7,3,2,3,6,6,5,1,5,6,7,1,4,7,3,6,6,3,8,8,6,8,4,5,3,8,7,1,5,8,6,4,8,2,8,5,3,5,3,1,8,6,4,4,8,3,4,3,3,1,8,4,8,5,3,5,8,6,6,4,4,4,7,1,3,8,6,5,1,5,3,6,6,4,7,1,2,1,7,5,6,5,3,1,1,4,2,2,3,3,1,3,2,1,1,4,4,4,8,1,5,7,2,6,8,8,8,4,4,2,6,2,1,6,7,3,5,3,7,2,3,6,8,6,3,3,3,2,8,7,2,5,4,3,1,8,1,3,8,3,7,8,2,1,7,2,1,7,6,8,6,8,8,2,8,3,3,4,8,5,8,8,8,5,7,7,2,8,1,8,1,5,4,7,8,4,4,6,1,7,5,6,6,2,2,3,7,5,6,4,1,6,7,5,3,7,3,4,2,8,8,2,3,7,7,2,5,1,1,7,8,5,6,3,3,2,3,4,4,1,8,2,5,7,1,3,1,5,1,5,5,6,6,5,2,5,6,4,2,7,7,8,3,4,6,4,2,3,2,6,1,5,6,3,5,6,5,2,4,4,5,3,2,8,6,6,3,4,5,8,7,1,8,4,5,2,3,6,1,8,2,2,5,1,4,7,4,8,7,1,7,8,3,8,5,6,2,7,4,1,6,6,3,3,7,8,7,8,3,5,6,7,1,7,1,1,7,6,2,2,1,8,5,5,7,4,5,6,5,4,6,6,8,4,7,2,8,3,8,7,5,2,1,8,7,5,3,4,3,8,6,8,8,6,5,2,5,8,2,4,6,4,3,8,7,6,5,7,7,5,4,8,5,1,2,7,5,1,4,4,2,4,7,3,7,2,8,4,3,1,8,1,4,3,6,1,4,4,5,5,4,5,6,3,8,1,2,8,2,6,8,3,5,4,5,1,5,8,7,4,2,4,6,7,3,7,1,3,1,1,4,3,5,8,5,2,4,3,6,5,6,5,6,2,4,1,1,3,5,7,8,3,1,4,4,6,7,6,6,7,2,4,1,5,1,4,1,2,1,6,1,2,8,6,5,4,3,8,7,6,3,1,7,2,6,5,6,5,5,6,7,8,1,8,3,8,8,2,3,2,8,3,2,8,4,8,2,4,1,6,7,8,7,5,1,6,3,5,3,3,4,2,5,2,1,5,4,3,8,7,8,6,2,2,1,3,7,5,1,3,7,8,3,1,3,3,7,7,5,7,7,7,8,5,7,2,4,4,1,6,5,4,7,6,1,8,3,7,6,3,4,6,1,6,6,3,4,7,7,3,7,5,2,6,3,6,8,8,2,4,7,7,8,4,6,3,6,3,4,4,4,2,5,6,6,5,3,1,1,5,3,4,3,4,5,2,3,3,3,5,5,2,2,3,7,6,5,2,2,8,7,1,4,3,8,5,6,5,1,8,7,4,8,1,2,6,8,8,2,5,2,4,2,3,8,6,3,3,8,7,2,7,5,6,4,1,8,8,7,4,4,4,4,1,6,4,7,1,5,6,5,8,2,8,1,6,7,4,7,4,5,3,8,3,2,1,8,7,7,4,5,8,6,7,6,6,1,8,6,1,7,6,3,5,5,7,4,4,8,3,7,2,2,3,6,6,3,1,1,2,1,7,1,3,2,3,2,1,7,7,7,8,6,7,3,7,7,7,6,6,1,2,4,8,4,4,7,3,3,5,8,3,2,5,3,6,8,6,1,4,4,1,7,1,4,4,8,1,3,8,8,7,6,2,2,8,4,1,8,1,6,3,8,5,2,5,4,3,1,1,6,6,2,7,5,2,4,5,6,1,7,1,4,1,1,5,6,8,4,4,5,5,3,3,8,8,7,7,8,8,3,7,3,1,8,4,6,7,8,2,2,2,3,6,2,5,2,7,8,8,6,1,7,8,8,3,2,5,1,3,1,2,7,7,4,7,8,6,7,7,4,2,7,4,1,5,8,7,4,1]
#f_memory = [1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,1,1,0,1,1,0,0,1,0,1,1,0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,1,1,1,1,0,1,1,0,0,0,1,0,1,0,1,1,1,0,0,0,1,1,1,0,1,1,0,0,1,1,0,1,1,0,1,1,1,1,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,0,1,0,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,0,0,1,1,1,0,1,1,0,1,1,1,0,1,0,0,0,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,1,1,0,1,1,0,1,1,1,0,1,0,0,0,0,1,0,1,1,1,1,0,0,1,0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,0,1,1,0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,1,0,1,1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,1,0,1,1,1,1,0,0,0,1,1,1,0,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,0,0,0,1,1,1,0,1,1,1,0,0,0,1,0,1,1,0,0,0,0,1,1,1,0,0,0,0,1,0,1,1,0,1,1,0,0,0,1,1,0,0,1,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,1,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,0,0,1,1,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,1,0,0,0,0,1,0,0,1,0,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,0,1,0,1,0,0,1,1,1,0,1,1,0,1,0,0,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,0,0,1,0,1,1,0,1,1,1,0,0,1,0,1,1,0,1,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,1,1,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,0,1,0,0,1,0,0,1,1,0,0,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,0,1,1,1,1,0,0,1,0,0,1,1,1,0,0,0,0,1,1,1,0,1,0,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,0,1,1,1,1,0,0,0,1,0,1,0,0,0,1,1,0,0,0,0,1,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,1,1,0,1,1,0,0,0,1,0,1,0,1,1,0,1,0,1,0,1,1,0,0,0,1,0,1,1,1,0,0,0,0,1,1,0,0,1,1,0,0,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1,0,1,0,0,0,0,1,1,0,1,0,0,0,0,0,1,0,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,1,0,1,1,0,0,0,1,0,0,1,1,0,1,0,0,1,0,1,1,0,0,1,0,0,1,1,0,1,0,1,1,0,0,1,0,0,1,0,1,1,1,1,0,0,1,1,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,0,1,1,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,1,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,1,0,1,0,1,1,1,0,1,1,1,1,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,1,1,0,1,0,0,1,1,0,1,1,1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,0,1,1,1,1,1,0,1,1,0,0,0,0,0,0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,1,0,0,1,0,1,1,0,1,0,1,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,1,1,1,1,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,1,1,1,1,0,0,1,0,1,0,0,0,1,1,0,1,0,1,0,1,0,0,1,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,1,0,1,1,1,1,0,1,0,1,1,0,0,1,0,0,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,0,1,1,0,0,0,1,0,1,1,0,0,1,0,0,1,0,0,0,1,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,0,0,0,1,1,1,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,0,1,1,1,1,0,1,0,0,1,0,1,0,1,0,0,0,1,1,0,0,1,1,1,1,0,0,0,0,1,1,0,0,1,0,0,1,1,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,0,1,1,0,0,1,0,1,1,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,0,1,1,1,0,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,0,1,1,0,0,1,1,0,0,0,1,0,1,1,0,1,1,0,0,1,1,1,0,1,1,0,0,1,1,1,1,0,0,1,1,1,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,0,1,0,1,1,1,0,0,0,1,1,1,0,1,1,0,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1,1,1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,1,1,1,0,1,1,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,0,0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,1,0,1,0,0,1,1,0,0,0,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,1,1,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,0,1,1,0,0,0,0,0,1,1,0,0,1,1,0,1,1,0,1,0,0,0,1,0,1,0,0,1,1,1,0,1,1,1,0,1,1,0,0,1,0,1,1,1,0,0,1,1,1,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,1,1,1,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,0,0,1,0,0,0,0,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,1,1,1,1,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,0,1,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,0,0,0,1,1,1,1,0,1,0,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,0,0,1,0,1,1,1,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,1,1,1,1,0,0,0,1,0,1,1,1,0,1,0,0,0,0,0,1,0,1,1,0,0,0,1,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0,1,1,0,1,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,1,1,0,0,1,0,0,0,1,0,1,0,0,1,1,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,0,0,0,0,1,0,1,1,1,1,0,0,0,1,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,0,1,0,1,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,1,1,1,0,1,0,0,1,0,1,0,0,0,0,0,1,1,1,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,1,1,0,0,1,0,1,1,1,0,0,1,1,1,0,0,1,0,0,0,0,0,1,1,0,0,0,1,0,0,1,1,0,1,1,0,1,1,1,1,0,1,0,0,1,0,1,0,1,0,1,1,0,0,0,0,1,1,0,1,1,0,1,1,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,1,1,1,1,0,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,1,1,0,1,0,1,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,1,0,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,0,1,1,1,1,1,1,0,0,0,1,1,1,0,1,1,0,0,1,1,1,1,1,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,1,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,1,1,1,1,0,1,1,0,0,1,0,0,1,0,0,0,1,1,0,1,0,0,1,1,0,1,1,1,0,0,1,1,1,0,0,0,0,1,0,1,0,1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,0,1,1,0,0,0,1,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,1,1,1,0,0,0,0,1,1,1,1,1,0,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,0,0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,0,0,0,0]
f_memory = [1, 0, 0, 0, 1, 1, 1, 1]
#f_obstacles = [0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,0,1,0,0,1,1,0,0,1,0,0,1,0,0,1,1,1,1,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,0,1,0,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,0,1,0,1,0,1,0,1,1,1,0,0,1,1,1,1,1,0,1,1,1,0,0,1,1,0,0,1,0,0,1,0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,0,0,0,1,0,1,1,1,1,1,0,0,0,0,1,1,0,0,1,0,0,0,1,1,1,0,0,1,0,1,1,1,1,1,0,1,0,1,0,0,1,0,1,0,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,0,0,0,0,1,1,1,0,1,0,0,1,0,0,1,0,0,1,1,1,0,0,0,1,1,0,1,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,1,1,1,0,1,0,1,1,1,1,0,0,1,1,1,0,0,1,1,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,1,0,1,1,1,0,0,1,1,0,0,0,0,0,1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,1,0,1,1,0,1,1,0,1,0,0,0,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,1,0,1,1,0,0,0,1,0,1,1,0,0,0,1,1,0,0,1,0,0,1,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,1,0,1,0,0,1,0,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,1,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,1,1,0,1,0,0,1,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0,1,0,1,1,0,1,0,1,0,1,1,1,0,0,1,0,0,0,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,0,1,0,1,1,0,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,1,1,1,0,0,0,0,1,1,0,1,0,1,1,0,1,1,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,1,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,1,1,0,1,1,0,0,0,0,1,1,0,1,0,0,0,1,1,1,0,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,0,1,1,1,1,0,1,1,0,0,1,0,0,1,0,1,1,1,0,0,1,1,0,1,0,1,0,0,1,1,1,1,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,0,0,0,1,1,1,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,0,0,1,0,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,0,0,1,1,0,0,1,0,0,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,0,0,0,0,1,0,1,1,1,0,1,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0,0,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,0,0,0,1,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,1,1,0,1,1,1,0,0,1,0,1,0,1,1,0,0,1,1,0,0,1,0,0,0,1,1,0,1,1,1,1,1,0,1,1,0,0,1,1,0,0,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,1,0,1,1,1,1,0,1,0,1,0,1,0,1,1,0,0,1,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1,0,0,0,1,0,0,1,1,0,0,1,0,1,0,0,1,1,0,0,0,0,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,1,1,0,0,0,1,0,1,1,0,0,1,1,0,0,0,0,1,1,0,0,0,1,1,1,0,1,0,1,1,1,0,1,0,0,1,0,1,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0,1,1,1,1,1,0,1,0,0,1,0,1,0,1,1,1,0,0,1,0,0,1,0,1,0,1,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,0,1,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,1,1,0,1,0,0,1,1,1,0,1,1,1,0,1,0,0,1,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,1,1,0,1,1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,0,0,1,0,0,1,1,0,0,1,0,0,1,1,1,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,1,0,0,0,0,1,0,1,0,0,0,1,1,1,1,0,1,1,0,0,1,0,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,0,0,1,0,1,1,1,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,0,0,1,0,1,0,0,1,1,0,1,1,1,1,1,1,1,0,1,0,0,1,0,1,1,0,1,1,0,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,0,1,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,1,0,0,0,1,1,0,1,0,0,1,0,1,1,0,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,0,0,1,1,1,0,1,1,0,1,1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,0,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,1,1,1,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,1,1,1,1,1,0,1,0,1,0,0,0,1,0,1,0,1,0,0,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,0,1,0,1,1,0,1,1,1,1,0,0,1,1,0,0,0,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,1,1,0,0,1,1,0,1,0,0,1,1,1,0,1,0,1,0,1,0,0,1,1,1,1,1,0,1,0,1,1,0,1,1,0,1,0,0,0,1,1,1,1,0,0,1,1,0,1,1,1,0,0,0,1,1,0,0,0,0,1,0,1,1,1,1,0,0,0,0,1,0,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,1,1,1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,0,0,0,1,0,0,1,1,0,1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,1,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,0,1,0,0,1,0,1,1,1,1,0,1,0,1,0,0,1,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,1,1,1,0,1,1,0,0,1,0,1,1,1,0,0,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,1,1,0,1,0,0,1,0,1,0,0,1,1,0,0,1,0,0,1,1,0,1,1,1,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,1,0,1,0,0,1,1,0,1,1,0,1,0,0,1,1,0,0,1,1,1,1,1,1,0,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,1,0,1,0,1,0,0,1,0,1,0,0,1,0,0,1,1,0,1,1,1,0,1,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,0,1,0,0,1,0,1,1,0,0,0,0,1,1,1,1,1,1,0,0,1,0,1,0,1,1,0,1,0,1,1,1,1,1,0,1,0,1,0,0,1,0,1,0,0,0,1,0,1,1,1,1,0,1,0,1,1,0,1,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,0,0,1,1,1,1,0,0,1,0,0,1,1,0,1,0,1,1,1,1,0,0,0,1,1,1,0,1,1,0,1,0,0,1,1,0,1,1,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,1,1,1,1,1,1,0,0,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,0,1,1,1,0,1,1,0,1,1,0,0,1,1,0,1,0,1,1,0,1,1,0,1,1,0,1,0,0,0,1,1,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,0,1,0,0,0,1,1,1,1,0,0,0,0,1,0,1,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,1,0,1,1,1,1,1,1,1,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1]
f_obstacles = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#f_area_size = [30,120,30,30,120,30,30,30,30,120,30,30,30,120,120,30,30,120,120,30,120,120,120,120,120,30,120,120,120,120,120,30,30,120,120,120,120,120,120,30,120,30,30,30,120,120,30,30,120,120,30,120,30,120,120,30,120,120,120,30,30,120,120,120,30,120,120,120,30,30,30,120,120,30,120,120,30,30,120,120,30,30,120,30,30,30,30,120,120,120,120,120,30,30,120,30,30,120,120,30,30,30,120,120,120,30,30,30,30,30,30,30,120,120,120,120,120,120,120,120,120,30,30,30,30,120,30,30,30,30,120,30,120,30,120,120,30,30,120,120,120,30,30,120,30,120,30,30,30,120,120,120,120,120,30,120,30,120,30,120,30,30,30,120,120,120,120,30,30,30,120,30,120,30,30,30,120,30,30,120,30,120,30,120,120,120,30,120,120,120,30,30,120,120,120,30,30,30,120,120,120,30,30,120,120,30,30,120,30,30,120,30,30,30,30,30,120,120,30,120,30,30,30,120,30,30,30,120,30,120,120,30,30,120,120,30,30,120,120,120,120,30,120,120,120,120,120,30,120,30,120,120,30,30,30,120,30,30,120,30,120,30,120,30,30,30,30,120,30,120,30,30,30,120,30,30,30,30,30,120,120,30,30,30,30,30,30,30,120,30,120,120,30,120,30,30,30,120,120,120,120,30,30,120,120,30,120,120,30,120,30,30,30,30,30,120,30,120,30,30,30,30,30,120,30,120,120,30,30,120,30,30,30,120,30,120,30,30,30,30,120,30,120,30,30,30,30,30,30,120,30,120,120,30,30,120,120,120,30,120,120,120,120,30,120,30,120,30,120,30,30,30,120,120,30,30,30,120,120,120,120,120,120,30,120,30,30,120,120,30,120,120,30,30,30,120,120,120,120,120,30,120,120,120,30,120,30,120,30,120,120,30,120,30,120,30,30,120,30,120,120,30,30,30,30,30,120,30,120,30,120,30,30,120,30,120,30,30,30,30,30,120,120,30,30,120,120,30,120,120,120,120,30,120,30,120,30,30,30,30,120,120,120,30,120,120,120,30,30,30,30,120,120,30,120,120,120,30,30,120,120,120,30,120,120,120,30,120,120,30,120,30,30,120,120,30,120,30,120,30,120,120,30,120,30,30,120,120,30,120,120,30,120,30,120,30,30,30,120,120,120,120,30,120,120,30,30,30,120,120,30,120,120,30,30,30,30,120,120,120,30,30,30,30,120,120,30,120,120,30,120,120,120,30,120,120,120,120,30,30,120,30,30,120,30,120,30,120,120,120,120,120,120,30,30,120,120,120,30,120,120,120,30,120,30,30,30,120,120,120,30,30,30,120,120,120,30,30,30,30,30,120,30,120,30,30,120,30,30,120,30,30,120,120,120,30,120,30,120,30,30,30,120,120,30,120,120,120,30,120,30,120,30,30,120,120,120,30,120,30,120,30,120,30,30,120,30,30,120,30,120,120,120,120,120,30,30,120,30,120,30,30,30,30,120,120,120,120,30,120,120,30,120,120,120,30,120,120,120,120,120,120,120,120,30,30,120,120,30,120,30,120,120,30,30,120,120,30,30,30,30,120,30,30,120,120,30,30,30,30,120,30,120,30,30,30,120,30,120,120,30,30,120,30,30,30,30,120,120,30,30,30,30,120,120,30,30,120,30,120,120,30,30,30,120,30,30,120,120,30,120,120,120,120,30,120,30,120,120,30,120,120,120,120,30,120,30,120,120,120,120,120,120,30,30,120,120,120,30,120,120,30,30,120,30,120,30,30,30,30,30,120,120,120,120,30,30,120,30,30,30,30,120,120,120,30,30,30,30,30,120,120,30,30,30,30,120,30,120,30,30,120,30,120,30,120,30,30,30,120,30,30,30,120,30,120,120,120,120,120,120,30,120,30,30,120,120,30,30,30,120,120,120,30,30,30,30,30,120,120,30,30,30,30,120,30,30,120,120,120,30,120,30,30,30,30,30,120,30,30,30,120,120,120,120,120,120,120,30,30,30,30,120,30,30,120,30,30,120,120,30,30,30,120,120,30,30,30,30,30,30,120,120,30,30,30,30,30,120,30,120,120,120,30,120,120,120,30,120,120,120,120,120,120,120,30,30,30,120,120,120,30,120,30,120,120,120,30,120,120,30,120,120,30,30,120,30,30,30,30,30,30,30,120,120,120,30,120,30,30,120,30,30,120,30,120,120,120,30,30,120,30,30,30,120,30,30,30,120,120,120,120,30,120,30,30,30,30,30,30,120,120,120,30,120,30,30,30,120,30,120,120,120,120,30,120,30,120,30,120,30,120,120,30,30,120,120,30,120,30,30,30,120,30,30,30,120,120,120,30,30,120,30,30,120,30,30,30,30,120,120,30,30,120,120,120,30,120,120,120,30,30,120,30,30,30,30,120,30,120,120,120,30,30,120,30,30,120,120,120,120,120,30,30,120,30,30,120,120,30,30,120,120,30,30,120,120,120,120,120,120,120,120,30,30,120,30,30,120,30,120,120,120,120,120,120,30,30,120,120,30,30,30,120,30,120,120,30,30,120,30,120,30,120,120,30,30,30,30,120,120,120,120,30,30,120,120,30,30,120,120,120,30,30,120,30,30,120,30,30,30,120,30,120,120,120,30,120,120,120,30,30,30,120,30,30,120,120,120,120,120,120,120,30,30,30,30,120,120,120,120,120,30,30,30,30,30,30,120,120,120,30,120,30,30,120,120,30,30,120,30,120,30,30,120,30,30,120,30,120,120,30,120,30,120,120,30,120,120,30,30,120,30,120,30,120,120,30,30,120,30,30,30,120,120,120,30,120,30,30,30,30,30,30,120,120,30,120,120,120,30,30,120,120,30,120,30,30,120,120,30,30,120,30,30,30,120,30,30,120,120,120,30,120,30,30,120,30,30,30,120,120,30,30,120,30,120,30,30,120,30,30,30,120,120,120,30,120,120,120,120,30,120,30,120,30,120,30,120,120,120,120,30,30,120,120,120,30,30,120,30,30,30,120,120,120,30,30,120,120,120,30,30,120,30,120,120,120,120,120,120,120,120,120,30,30,120,120,30,120,120,30,120,30,30,120,120,30,30,30,120,120,120,120,120,30,30,30,30,120,120,30,30,120,30,30,120,120,120,120,120,30,30,120,30,30,30,120,30,30,120,30,30,120,120,30,120,120,120,120,30,30,120,30,30,30,30,120,120,30,120,30,30,30,30,120,30,30,120,120,30,120,120,120,120,30,120,120,30,30,30,30,120,120,30,30,120,30,30,120,30,120,120,120,30,30,120,120,120,120,30,30,30,120,120,30,120,120,30,30,120,120,30,30,30,30,120,120,120,120,120,120,120,120,120,120,30,30,120,30,30,30,30,30,30,30,30,30,120,120,120,120,120,120,30,120,120,30,30,120,30,120,120,30,30,120,120,30,120,30,30,30,120,30,120,30,30,120,30,120,120,120,120,120,120,30,30,120,120,30,120,120,30,30,120,120,30,30,30,30,120,120,30,30,30,120,120,120,30,120,30,120,30,30,30,120,30,120,30,120,120,120,30,120,120,30,120,30,30,30,120,120,120,30,30,30,30,120,120,30,30,30,120,30,120,120,120,30,30,120,30,120,120,120,120,30,30,120,30,30,120,30,120,120,30,120,30,120,30,120,120,30,120,30,30,120,30,30,120,30,120,30,30,120,120,120,30,120,120,30,120,120,30,30,120,30,120,30,120,120,120,30,30,30,30,30,120,30,120,30,120,120,30,30,120,30,120,30,120,30,30,30,30,120,30,30,120,120,30,30,30,120,30,30,30,120,120,120,120,30,30,120,30,120,30,120,30,120,30,120,30,120,30,120,30,30,120,30,30,30,120,30,30,120,30,30,30,30,30,120,30,30,30,30,120,30,30,30,120,120,120,30,30,120,30,120,120,30,120,120,120,30,120,120,30,120,30,30,120,120,120,120,120,30,30,30,30,30,120,30,30,30,120,30,30,120,30,30,120,120,30,120,30,120,30,120,30,30,120,30,120,30,30,120,120,120,120,120,120,30,30,30,120,30,120,120,120,30,30,30,120,120,30,30,120,120,120,120,30,30,120,120,30,30,30,30,30,120,30,120,30,120,30,120,120,30,30,120,30,30,120,30,120,120,30,120,30,120,120,30,120,120,30,120,30,30,30,120,120,120,120,30,30,30,30,120,30,120,30,120,30,30,30,120,30,30,30,30,120,120,120,120,120,30,120,30,30,120,120,120,120,120,30,120,30,30,120,30,120,120,120,120,30,120,120,120,30,120,120,120,30,30,30,120,30,120,30,120,30,30,120,30,30,30,120,120,120,30,30,120,120,120,30,120,120,120,30,120,30,30,120,30,30,30,30,30,120,120,120,30,120,120,30,30,30,30,120,30,30,120,120,120,120,30,30,30,30,120,30,120,30,30,30,120,30,30,120,120,30,120,120,120,120,30,30,120,120,30,120,120,30,120,30,120,120,30,30,30,120,120,120,30,30,30,120,30,30,120,30,30,120,120,30,30,120,120,120,120,120,120,30,120,120,30,30,30,120,120,120,30,120,30,30,120,30,120,120,30,30,120,30,30,120,120,30,30,30,120,30,30,30,30,120,30,120,30,120,120,30,120,30,30,30,120,30,120,30,120,120,120,120,120,120,30,30,120,30,30,30,120,120,30,30,30,30,30,120,120,30,30,120,30,30,30,120,30,120,30,120,30,30,120,30,30,120,120,120,120,30,120,120,120,120,120,30,120,30,30,30,30,30,30,120,30,30,120,30,120,120,30,120,30,30,30,120,30,30,30,30,30,120,120,120,120,30,120,120,30,30,30,120,120,120,120,30,120,120,120,120,120,120,120,30,120,30,120,120,30,30,30,30,120,30,120,120,120,120,120,30,30,30,30,30,30,30,30,30,30,30,120,30,30,120,120,30,120,120,120,30,120,30,30,30,120,30,30,30,30,30,30,120,120,120,30,120,30,120,30,30,30,120,120,30,120,120,30,30,120,30,120,120,120,30,30,120,30,30,30,30,120,30,30,120,120,30,30,30,30,120,30,30,120,30,120,30,30,30,30,30,120,120,120,120,120,30,120,120,120,30,120,120,30,30,120,120,30,120,30,120,120,120,30,30,120,30,30,120,120,120,30,30,30,120,30,120,30,30,30,30,120,120,120,30,30,120,30,30,30,120,30,30,30,120,120,30,120,30,30,30,120,120,120,30,30,120,30,30,30,120,30,120,30,120,30,30,120,30,120,120,30,120,120,120,30,30,30,120,120,30,120,30,30,30,30,30,30,120,30,30,30,30,30,120,120,30,30,30,120,30,120,30,120,120,120,120,30,120,120,30,120,120,120,30,120,120,30,120,30,120,30,30,30,30,30,120,120,120,30,120,120,120,30,30,120,30,120,120,30,120,120,30,120,30,30,30,120,120,120,30,30,30,120,30,120,30,120,120,30,30,30,30,120,30,120,30,120,30,120,120,120,120,30,30,30,30,30,30,30,30,120,120,120,120,30,30,30,30,30,120,30,30,30,30,120,30,30,120,30,30,30,120,30,30,30,30,30,30,120,120,30,30,120,120,120,120,30,30,30,120,120,120,120,120,30,120,30,30,120,120,30,120,30,120,120,30,30,120,30,120,120,120,30,120,30,120,30,120,120,30,120,30,30,30,120,30,30,30,120,30,30,30,120,30,30,120,30,120,30,30,30,120,120,30,30,120,120,120,30,30,120,30,30,120,120,30,120,120,30,120,120,120,30,120,30,120,120,120,30,30,120,120,120,30,120,30,120,30,30,30,120,120,120,30,30,120,120,30,120,120,120,120,120,30,30,30,120,120,120,120,120,120,30,120,120,120,120,120,120,120,120,120,120,120,120,30,30,30,120,30,30,30,120,120,120,120,120,30,120,120,30,30,30,30,30,120,120,120,120,30,120,30,30,120,120,120,120,30,120,120,120,120,120,120,30,30,30,120,30,120,120,120,120,120,120,30,30,120,120,120,30,120,30,120,120,30,30,120,120,120,30,120,120,30,30,30,30,30,120,120,120,120,120,30,30,30,30,120,30,30,30,120,30,30,120,120,120,30,120,120,30,120,30,120,120,30,30,30,30,120,120,30,30,30,30,120,120,120,30,120,30,30,30,30,30,120,30,120,30,30,30,30,30,120,120,120,120,30,30,30,30,120,120,30,120,120,30,30,120,120,30,120,30,30,30,30,30,120,30,30,120,120,30,30,120,120,120,30,120,120,120,120,30,120,120,120,120,30,120,30,30,30,30,30,30,120,120,30,120,120,30,30,120,120,120,30,120,30,30,30,120,30,30,30,30,120,30,30,30,30,30,120,30,30,120,120,120,120,120,120,30,120,120,30,120,120,30,120,30,120,30,30,30,30,30,120,120,30,120,30,30,120,120,120,30,30,120,120,30,120,120,30,120,30,120,120,120,120,120,120,120,120,120,120,30,30,120,120,120,30,120,30,120,120,120,120,30,120,30,30,30,120,120,30,120,30,30,120,30,30,120,30,120,30,120,120,30,120,30,120,120,120,30,30,30,120,30,30,30,120,120,30,30,30,30,120,30,120,30,30,30,120,120,30,120,120,30,120,120,30,120,30,120,30,30,120,120,120,120,120,120,120,120,120,120,30,30,120,30,120,120,30,120,120,30,120,30,30,120,30,120,30,30,120,30,120,120,120,30,120,30,120,30,30,120,120,30,120,30,120,30,30,120,120,30,30,120,120,30,30,120,120,30,120,30,30,120,120,120,30,30,30,120,120,120,120,30,120,120,30,30,120,120,120,30,120,30,120,30,120,30,30,120,30,30,120,120,30,30,30,30,120,120,30,120,120,120,120,30,30,30,120,120,120,30,120,30,120,30,120,30,30,120,30,120,120,120,30,120,30,30,120,30,30,120,30,30,120,120,30,30,120,120,30,30,30,120,30,120,120,30,120,120,30,120,120,120,120,30,120,120,30,120,30,120,120,120,120,120,30,30,120,30,30,120,30,120,30,120,120,30,120,30,120,30,120,120,30,120,120,30,120,30,30,30,120,120,120,30,120,30,30,120,120,30,30,30,30,120,120,120,30,30,30,30,30,120,120,30,120,120,30,30,120,120,120,30,30,120,30,120,120,30,30,120,30,120,30,120,120,120,120,120,120,120,120,120,30,120,120,30,30,30,30,120,30,30,30,30,120,120,30,120,30,30,30,120,120,120,120,30]
f_area_size = [30, 30, 30, 30, 30, 30, 30, 30]
for i in range(runs):
    j = i % 3200
    #p_s_space = 35
    p_s_space = f_area_size[j]
    if f_area_size[j] == 120:
        o_m_size = 2
        p_endurance = 100000000000000000/2
    else:
        o_m_size = 25
        p_endurance = 100000000000000000/2
    #run(z=i, func_number=7, n=9, n_obstacles=f_obstacles[i], memory=0)
    run(z=i, func_number=f_numbers[i], n=9, n_obstacles=f_obstacles[i], memory=2)
    # cProfile.run('run(z=i, func_number=f_numbers[i], n=9, n_obstacles=f_obstacles[i], memory=f_memory[i])')
    if i % 800 == 0:
        data_df = pd.DataFrame(g_data)
        data_df.to_csv(str(i) + str(today)+'phase_4.csv')
data_df = pd.DataFrame(g_data)
data_df.to_csv(str(today)+'phase_4_all.csv')

end = time.time()

