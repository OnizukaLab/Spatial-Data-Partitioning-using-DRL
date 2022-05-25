# -*- coding: utf-8 -*-
from cmath import exp
import numpy as np
import pandas as pd
import random
import numpy as np
import math
from collections import namedtuple

import warnings

from torch.nn.modules import loss
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from Model import Net
from ReplayMemory import ReplayMemory, PERMemory
from utils import *

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'is_demo', 'legal_actions', 'next_legal_actions'))

class Agent:
    def __init__(self, num_states, num_actions, params, env):
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.params = params
        self.env = env

        self.use_per = params["use_per"]
        self.agent_memory = PERMemory(params["agent_memory_capacity"]) if self.use_per else ReplayMemory(params["agent_memory_capacity"])
        self.demo_memory = PERMemory(params["demo_memory_capacity"]) if self.use_per else ReplayMemory(params["demo_memory_capacity"])
        self.multi_step_transitions = []
        self.one_episode_transitions = []

        self.main_q_network = Net(num_states, num_actions)
        self.target_q_network= Net(num_states, num_actions) 
        print(self.main_q_network) 

        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=params["lr"])
        self.criterion = nn.MSELoss()
    
    def get_legal_actions(self, state):
        state_ = state.numpy()

        legal_actions = []

        for action in range(self.num_actions):
            action_ = self.form_action(action)

            if (action_[2] == 0 and state_[0, action_[0], action_[1]] == 0 and state_[1, action_[0], action_[1]] == 1) or (action_[2] == 1 and state_[0, action_[0], action_[1]] == 1 and state_[1, action_[0], action_[1]] == 0):
                    legal_actions.append(True)      
            else:
                legal_actions.append(False)

        return np.array(legal_actions)
    
    def get_action(self, state, legal_actions, eps_random=0, eps_shift=0, test=False):
        self.main_q_network.eval() 

        legal_actions_idx = [i for i, x in enumerate(legal_actions) if x == True]

        # ε-greedy
        r = random.random()
        if not test:
            if r < eps_random: # random action
                actions = [idx for idx, legal in enumerate(legal_actions) if legal]
                action = random.choice(actions)
            else: # max action
                with torch.no_grad():
                    qs = self.main_q_network(state).squeeze()
                    legal_actions = torch.Tensor(legal_actions).type(torch.LongTensor)
                    legal_qs = qs[legal_actions==True]
                    action = legal_actions_idx[torch.argmax(legal_qs)]

                # ---- grid-shift action -----
                if r > 1-eps_shift:
                    action_ = self.form_action(action)
                    shift_actions = [s for s in range(-2*self.params["shift_grids"], 2*(self.params["shift_grids"]+1), 2) if s != 0]
                    eps_width = eps_shift / (2*self.params["shift_grids"])
                    shift_choice = int((r-(1-eps_shift)) / eps_width)
                    
                    if action_[2] == 0:
                        grid_shift_action = action + shift_actions[shift_choice]
                    else:
                        grid_shift_action = action + shift_actions[shift_choice]*self.env.grids["long_num"]
                    
                    try:
                        action=legal_actions_idx[legal_actions_idx.index(grid_shift_action)]
                    except:
                        pass 
        else:
            with torch.no_grad():
                qs = self.main_q_network(state).squeeze()
                legal_qs = qs[legal_actions==True]
                action = legal_actions_idx[torch.argmax(legal_qs)]

        return action
    
    def form_action(self, action):
        t = 2*self.env.grids["long_num"]
        i = int(action / t)
        j = int((action - i*t) / 2)
        d = int(action % 2)
        action_ = np.array([i, j, d], dtype=int)

        return action_
    
    def add_memory(self, state, action, state_next, reward, is_demo, legal_actions):
        transition = Transition(state, action, state_next, reward, is_demo, legal_actions, None)

        if 1 < self.params["num_multi_step_reward"]:
            transition = self._get_multi_step_transition(transition)

        if transition == None:
            return
        
        if transition.is_demo:
            self.demo_memory.push(transition)
        else:
            self.agent_memory.push(transition)
        
    def _get_multi_step_transition(self, transition):
        self.multi_step_transitions.append(transition)
        
        if len(self.multi_step_transitions) < self.params["num_multi_step_reward"]:
            return None

        next_state = transition.next_state
        nstep_reward = 0
        for i in range(self.params["num_multi_step_reward"]):
            r = self.multi_step_transitions[i].reward
            nstep_reward += r * self.params["gamma"] ** i

            if self.multi_step_transitions[i].next_state is None:
                next_state = None
                break
            
        next_legal_actions = None
        if next_state is not None:
            next_legal_actions = self.get_legal_actions(next_state.squeeze())
            next_legal_actions = torch.from_numpy(next_legal_actions).type(torch.FloatTensor)

        state, action, _, _, is_demo, legal_actions, _ = self.multi_step_transitions.pop(0)
    
        return Transition(state, action, next_state, nstep_reward, is_demo, legal_actions, next_legal_actions)
    
    def optimize_model(self, mode, episode, num_episodes):
        '''Experience Replay'''
        if mode == "Train":
            if len(self.agent_memory) < self.params["batch_size"]:
                return

            demo_batch_size = int(self.params["batch_size"] * self.params["demo_ratio"])
            if len(self.demo_memory) < demo_batch_size:
                demo_batch_size = len(self.demo_memory)
            agent_batch_size = self.params["batch_size"] - demo_batch_size

            indexes, transitions, weights, memory_types = [], [], [], []
            # demo_memory
            if len(self.demo_memory) != 0:
                i, t, w = self.demo_memory.sample(demo_batch_size, episode, num_episodes)
                indexes.extend(i)
                transitions.extend(t)
                weights.extend(w)
                memory_types.extend([0 for _ in range(demo_batch_size)])
            # agent_memory
            i, t, w = self.agent_memory.sample(agent_batch_size, episode, num_episodes)
            indexes.extend(i)
            transitions.extend(t)
            weights.extend(w)
            memory_types.extend([1 for _ in range(agent_batch_size)])
            weights = np.array(weights)
        else:
            indexes, transitions, weights = self.demo_memory.sample(self.params["batch_size"], episode, num_episodes)
            memory_types = [0 for _ in range(self.params["batch_size"])]

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        is_demo_mask = torch.ByteTensor(tuple(map(lambda s: s is True, batch.is_demo)))


        legal_actions_mask= torch.cat([s for s in batch.legal_actions if s is not None]).type(torch.LongTensor)
        next_legal_actions_mask = torch.cat([s for s in batch.next_legal_actions if s is not None]).type(torch.LongTensor)

        # ==================== (1) N step TD error =================== '''
        self.main_q_network.eval()
        self.target_q_network.eval()

        qs = self.main_q_network(state_batch)
        qs[legal_actions_mask] = 0 
        selected_q = qs.gather(1, action_batch)

        qs_next_states = self.main_q_network(non_final_next_states)
        qs_next_states[next_legal_actions_mask] = 0 

        a_max_next_states = torch.zeros(self.params["batch_size"]).type(torch.LongTensor)
        a_max_next_states[non_final_mask] = qs_next_states.detach().max(1)[1]
   
        a_max_non_final_next_states = a_max_next_states[non_final_mask].view(-1, 1)
        
        target_q_next_state = torch.zeros(self.params["batch_size"])
        target_q_next_state[non_final_mask] = self.target_q_network(non_final_next_states).gather(1, a_max_non_final_next_states).detach().squeeze()

        expected_q = reward_batch + self.params["gamma"] * target_q_next_state
   
        n_td_error = ((selected_q - expected_q.unsqueeze(1)) ** 2) * torch.from_numpy(weights).unsqueeze(1)
        loss_n = self.params["lambda1"] * n_td_error.mean()

       
        # ========== (2) large margin classification loss=========
        q_max_info = torch.max(qs, 1) 
        q_max = q_max_info.values[is_demo_mask]
        a_max = q_max_info.indices[is_demo_mask] 
        q_demo = selected_q[is_demo_mask] 
        a_demo = action_batch[is_demo_mask].squeeze()

        non_match_idx = torch.where(a_max != a_demo) 
        margin_loss = q_max
        margin_loss[non_match_idx] += self.params["margin"] 

        lm_error = torch.abs(margin_loss.unsqueeze(1) - q_demo) * torch.from_numpy(weights)[is_demo_mask].unsqueeze(1)
        loss_E = self.params["lambda2"] * lm_error.mean()
        if math.isnan(loss_E):
            loss_E = torch.tensor([0])
       
        # ========== (3) L2 loss=========
        l2 = torch.tensor(0., requires_grad=True)
        for w in self.main_q_network.parameters():
            l2 = l2 + torch.norm(w)**2
        loss_l2 = self.params["lambda3"] * l2
        
        loss_all = loss_n + loss_E + loss_l2

        self.main_q_network.train()
        update_params(self.optimizer, loss_all)

        if (indexes != None):
            for i, value in enumerate(selected_q):
                td_error = abs(expected_q[i].item() - value.item())
                if memory_types[i] == 0:
                    self.demo_memory.update(indexes[i], td_error)
                else:
                    self.agent_memory.update(indexes[i], td_error)

        return (loss_n, loss_E, loss_l2, loss_all)