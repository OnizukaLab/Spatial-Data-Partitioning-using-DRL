# -*- coding: utf-8 -*-
import numpy as np
import gym
import cv2
import statistics
from numpy.core.numeric import normalize_axis_tuple
import pandas as pd
import copy
import csv
import sys
import math
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Environment for Spatial Partitioning
class SDPEnv(gym.Env):

    def __init__(self, env):
        self.points = pd.read_csv(env["points_path"], header=None)
        self.workload = env["workload"]
        print("{} records:{}".format(env["points_path"], len(self.points)), flush=True)
        print("workload:{}".format(self.workload), flush=True)

        self.run_args = env["run_args"]
        
        self.grids = env["grids"]
        self.num_machines = env["num_machines"]
        self.max_partitions = env["max_partitions"]
        self.is_run_query = env["is_run_query"]
        
        self.WINDOW_SIZE = 500

        self.state_conponents = 3
        self.num_states = (self.grids["lat_num"] + 1) * (self.grids["long_num"] + 1) * self.state_conponents
        self.num_actions = self.grids["lat_num"] * self.grids["long_num"] * 2

        self.state_shape = (self.state_conponents, self.grids["lat_num"]+1, self.grids["long_num"]+1)
        self.action_shape = (self.grids["lat_num"], self.grids["long_num"], 2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.state_shape) 
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.points_sub = self.points
        self.len_point = len(self.points_sub)
        self.query = self.workload["distance"]
        write_file(self.query, './tmp/query.csv', 'query')

        self.map = {'min_long': 0, 'max_long': 0, 'min_lat': 0, 'max_lat': 0}
        self.coordinates = {'lat': [], 'long': []} 

        decimals = 5
        self.map["min_long"] = RoundOff(self.points_sub[0].min(), decimals, False)
        self.map["max_long"] = RoundOff(self.points_sub[0].max(), decimals, True)
        self.map["min_lat"] = RoundOff(self.points_sub[1].min(), decimals, False)
        self.map["max_lat"] = RoundOff(self.points_sub[1].max(), decimals, False)

        self.coordinates["long"] = list(np.linspace(self.map["min_long"], self.map["max_long"], self.grids["long_num"]+1))
        self.coordinates["lat"] = list(np.linspace(self.map["min_lat"], self.map["max_lat"], self.grids["lat_num"]+1))

        self.cells =  {
            'pid': np.zeros((self.max_partitions, self.grids["lat_num"] + 1, self.grids["long_num"] + 1)),
            'dist': np.zeros((self.grids["lat_num"] + 1, self.grids["long_num"] + 1)) 
        } 

        for i in range(self.grids["lat_num"]):
            for j in range(self.grids["long_num"]):
                count = ((self.points_sub[0] >= self.coordinates["long"][j]) & (self.points_sub[0] <= self.coordinates["long"][j+1]) & \
                    (self.points_sub[1] >= self.coordinates["lat"][i]) & (self.points_sub[1] <= self.coordinates["lat"][i+1])).sum()
                self.cells["dist"][i, j] = count / self.len_point
        
        M = 1
        m = 0
        dist_min = self.cells["dist"].min(axis=None, keepdims=True)
        dist_max = self.cells["dist"].max(axis=None, keepdims=True)
        self.cells["dist"] = (self.cells["dist"] - dist_min) / (dist_max - dist_min) * (M - m) + m

        self.best_cost = 0 
        self.best_cost_sub = 0
        self.all_query_runtime = 0
         
    def reset(self, mode=None):
        self.mode = mode

        self.partitions = {'id': [], 'coordinates': [], 'data':[]}
        
        observation  = np.zeros(self.state_shape)
        
        for i in range(self.grids["long_num"]):
            observation[0, 0, i] = 1
            observation[0, self.grids["lat_num"], i] = 1

        for i in range(self.grids["lat_num"]):
            observation[1, i, 0] = 1
            observation[1, i, self.grids["long_num"]] = 1
        
        for i in range(self.grids["lat_num"]):
            for j in range(self.grids["long_num"]): 
                self.cells["pid"][0, i, j] = 1

        observation[2] = self.cells["dist"] 

        self.partitions["id"].append(0) 
        self.partitions["coordinates"].append([self.map["min_long"], self.map["max_long"], self.map["min_lat"], self.map["max_lat"]])
        self.partitions["data"].append(1.0) 
        
        self.observation_before = copy.deepcopy(observation)   

        return observation
    
    def get_demo_action(self):

        par_info = {"num_data":[], "lat_median": [], "long_median": []}
        for i, p in enumerate(self.partitions["coordinates"]):
            par_data = self.points_sub[(self.points_sub[0] >= p[0]) & (self.points_sub[0] <= p[1]) & (self.points_sub[1] >= p[2]) & (self.points_sub[1] <= p[3])] #(※計算時間がデータサイズ・パーティション数に準拠)
            par_info["num_data"].append(len(par_data[0]))
            try:
                par_info["long_median"].append(statistics.median(par_data[0].values.tolist()))
                par_info["lat_median"].append(statistics.median(par_data[1].values.tolist()))
            except statistics.StatisticsError:
                par_info["long_median"].append(0)
                par_info["lat_median"].append(0)
        idx_max = np.argmax(par_info["num_data"])
        
        long_width = self.partitions["coordinates"][idx_max][1] - self.partitions["coordinates"][idx_max][0]
        lat_width = self.partitions["coordinates"][idx_max][3] - self.partitions["coordinates"][idx_max][2]

        if lat_width > long_width:
            glines = self.coordinates["lat"][self.coordinates["lat"].index(self.partitions["coordinates"][idx_max][2])+1 : self.coordinates["lat"].index(self.partitions["coordinates"][idx_max][3])]
                
            if not glines:
                print("Reccomend adding more grids.", flush=True)
                return
            else:
                long_idx = self.coordinates["long"].index(self.partitions["coordinates"][idx_max][0])
                lat_idx = self.coordinates["lat"].index(glines[np.argmin(np.abs(np.array(glines) - par_info["lat_median"][idx_max]))]) 
                direction = 0

        else:
            glines = self.coordinates["long"][self.coordinates["long"].index(self.partitions["coordinates"][idx_max][0])+1 : self.coordinates["long"].index(self.partitions["coordinates"][idx_max][1])]

            if not glines:
                print("Reccomend adding more grids.")
                return
            else:
                lat_idx = self.coordinates["lat"].index(self.partitions["coordinates"][idx_max][2])
                long_idx = self.coordinates["long"].index(glines[np.argmin(np.abs(np.array(glines) - par_info["long_median"][idx_max]))]) 
                direction = 1

        act_ = [lat_idx, long_idx, direction]
        act = 2 * (act_[0] * self.grids["long_num"] + act_[1]) + act_[2]

        return act

    def step(self, action_):
        reward = 0
        done = False

        observation = copy.deepcopy(self.observation_before)
        
        p_onehot = list(self.cells["pid"][0:self.max_partitions, action_[0], action_[1]])
        pid_slct = p_onehot.index(1)
        pid_add = len(self.partitions["id"])
        lat = self.coordinates["lat"]
        long = self.coordinates["long"]        
        par_tmp = copy.deepcopy(self.partitions["coordinates"][pid_slct])

        self.partitions["id"].append(pid_add)

        w = 0
        i = action_[0]
        j = action_[1]
        if action_[2] == 0:
            while j < self.grids["long_num"]:
                observation[0, i, j+w] = 1
                if observation[1, i, j+w+1] == 1:
                    break
                w += 1
            
            border = lat[action_[0]]
            self.partitions["coordinates"].append([par_tmp[0], par_tmp[1], border, par_tmp[3]])

            self.partitions["coordinates"][pid_slct][3] = border
            updated_partition_idx = {
                "updated":[long.index(par_tmp[0]), long.index(par_tmp[1]), lat.index(par_tmp[2]), lat.index(border)],
                "added":[long.index(par_tmp[0]), long.index(par_tmp[1]), lat.index(border), lat.index(par_tmp[3])]
            }

        else:
            while i < self.grids["lat_num"]:
                observation[1, i+w, j] = 1
                if observation[0, i+w+1, j] == 1:
                    break
                w += 1

            border = long[action_[1]]
            self.partitions["coordinates"].append([border, par_tmp[1], par_tmp[2], par_tmp[3]])
            
            self.partitions["coordinates"][pid_slct][1] = border
            updated_partition_idx = {
                "updated":[long.index(par_tmp[0]), long.index(border), lat.index(par_tmp[2]), lat.index(par_tmp[3])],
                "added":[long.index(border), long.index(par_tmp[1]), lat.index(par_tmp[2]), lat.index(par_tmp[3])]
            }

        self.partitions["data"].append(0.0)

        for i, par in enumerate(self.partitions["coordinates"]):
            if i == pid_slct or i == pid_add:
                count = ((self.points_sub[0] >= par[0]) & (self.points_sub[0] <= par[1]) & \
                    (self.points_sub[1] >= par[2]) & (self.points_sub[1] <= par[3])).sum()
                self.partitions["data"][i] = count / self.len_point

        for p in updated_partition_idx:
            for i in range(updated_partition_idx[p][2], updated_partition_idx[p][3]):
                for j in range(updated_partition_idx[p][0], updated_partition_idx[p][1]):
                    if p == "updated":  
                        observation[2, i, j] = self.cells["dist"][i][j] * self.partitions["data"][pid_slct]
                    else:
                        observation[2, i, j] = self.cells["dist"][i][j] * self.partitions["data"][pid_add]
                        self.cells["pid"][pid_slct, i, j] = 0
                        self.cells["pid"][pid_add, i, j] = 1

        write_file(self.partitions["coordinates"], './tmp/partitions.csv', 'partitions') 
             
        if len(self.partitions["coordinates"]) >= self.max_partitions:
            if self.mode == "Train" or self.mode == "Test" :
                reward = self.get_reward()
            elif self.mode == "Demo":
                reward = 1.0
            else:
                reward = None
            done = True

        self.observation_before = copy.deepcopy(observation)

        return observation, reward, done

    def get_reward(self):

        if self.mode == "Test":
            limit_costs = [100000 for _ in range(len(self.query))]
            write_file(limit_costs, './tmp/run_cost.csv', 'query')
            write_file(limit_costs, './tmp/best_cost.csv', 'query')

            self.run_query(self.run_args)
            costs = read_file('./tmp/run_cost.csv')
            write_file(costs, './tmp/best_cost.csv', 'query')
            write_file(self.partitions["coordinates"], './tmp/best_partitions.csv', 'partitions')
            sum_cost = sum(np.array(costs)*np.array(self.workload["rate"]))
            self.best_cost = sum_cost
        else:
            self.run_query(self.run_args)
            costs = read_file('./tmp/run_cost.csv')
            if not costs:
                reward = 0.2
            else:
                sum_cost = sum(np.array(costs)*np.array(self.workload["rate"]))
                reward = (self.best_cost / sum_cost) ** 2

            if reward > 1:
                self.best_cost = sum_cost
                write_file(self.partitions["coordinates"], './tmp/best_partitions.csv', 'partitions')
                write_file(costs, './tmp/best_cost.csv', 'query')

            return reward
    
    def run_query(self, args):
        t0 = time.time()
        if self.is_run_query:
            subprocess.run(args, shell=False)
        t1 = time.time()
        query_time = t1 - t0
        self.all_query_runtime += query_time

    def render(self, path=None):
        fig = plt.figure()
        ax = plt.axes()

        for p in self.partitions["coordinates"]:
            r = patches.Rectangle(xy=(p[0], p[2]), width=p[1]-p[0], height=p[3]-p[2], ec='#000000', fill=False,  linewidth='0.8')
            ax.add_patch(r)

        ax.scatter(self.points_sub[0], self.points_sub[1], c='r', s=0.01)
        plt.axis('scaled')
        ax.set_aspect('equal')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        if path != None:
            plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
    

def RoundOff(x, n, s):
    if s: 
        y = math.ceil(x * 10 ** n) / (10 ** n)
    else:
        y = math.floor(x * 10 ** n) / (10 ** n)
    return y


def write_file(data, path, s):
    if s == "query":
        with open(path, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            for d in data:
                writer.writerow([d])
    else:
        with open(path, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(data)

def read_file(path):
    with open(path, 'r') as f:
        line = f.read().split()
        data = [float(d) for d in line]
    return data
