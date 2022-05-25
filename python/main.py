# -*- coding: utf-8 -*-
import datetime
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from Environment import SDPEnv
from Agent import Agent
from utils import *


class Environment:
    def __init__(self, config):
        self.params = config["params"]

        print("use_per:", self.params["use_per"], flush=True)
        print("num multi step:", self.params["num_multi_step_reward"], flush=True)

        self.env = SDPEnv(config["env"]) 
        self.num_states = self.env.observation_space.low.size
        self.num_actions = self.env.action_space.n
        self.agent = Agent(self.num_states, self.num_actions, self.params, self.env)

        self.demo_data = []

    def make_demo(self, mode="Demo"):
        total_step = 0

        print("----- make demo start -----")
        for episode in range(self.params["demo_episodes"]):
            done = False 

            observation = self.env.reset(mode) 
            state = torch.from_numpy(observation).type(torch.FloatTensor) 
            state = torch.unsqueeze(state, 0)
            
            action_list = []

            for step in range(self.params["max_steps"]): 

                legal_actions = self.agent.get_legal_actions(state.squeeze())
                action = self.env.get_demo_action()
                if action is None:
                    break
                else:
                    action = torch.LongTensor([[action]])
                action_list.append(action.item())
                legal_actions = torch.from_numpy(legal_actions).type(torch.FloatTensor)

                action_ = self.agent.form_action(action)
                observation_next, reward, done = self.env.step(action_)
                
    
                if done:
                    state_next = None
                else:
                    state_next = torch.from_numpy(observation_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                reward = torch.FloatTensor([reward])

                self.demo_data.append([state, action, state_next, reward, legal_actions])
                self.agent.add_memory(state, action, state_next, reward, True, legal_actions)
     
                state = state_next
                total_step += 1

                if done:
                    print("epsiode:{} action:{} reward:{}".format(episode+1, action_list, reward.item()), flush=True)
                    break

        print("---------------- finish----------------", flush=True)
        print("Total Time (only Run Query):", datetime.timedelta(seconds=self.env.all_query_runtime), flush=True)
        print("collected demo data size:", len(self.demo_data), flush=True)
        save_memory(self.demo_data, log_dir+'demo_data.pickle')

    def pretrain(self, mode="PreTrain"):

        logs = {
            "episode_reward": [],
            "episode_loss_n": [], 
            "episode_loss_E": [],
            "episode_loss_l2": [],
            "episode_loss_all": [],
            "best_cost": [] 
        }

        correct = 0
        self.test(mode=mode)
        print("----- PreTrain Start -----")
        start_time = time.time()
        self.inter = start_time 
        for episode in range(self.params["pre_num_episodes"]):
            loss = self.agent.optimize_model(mode, episode, self.params["pre_num_episodes"])
            logs["episode_reward"].append(0)
            logs["episode_loss_n"].append(loss[0].item())
            logs["episode_loss_E"].append(loss[1].item())
            logs["episode_loss_l2"].append(loss[2].item())
            logs["episode_loss_all"].append(loss[3].item())
            logs["best_cost"].append(self.env.best_cost)
        
            if episode % self.params["pre_update_target_freq"] == 0:
                hard_update_target_network(self.agent.main_q_network, self.agent.target_q_network)
                
            if (episode+1) % self.params["test_freq"] == 0:
                is_correct = self.test(mode=mode, policy=["pred"])
                if is_correct:
                    correct += 1
                    save_model(self.agent.main_q_network, model_dir+str(mode)+'_DQN.pth')
                    print(["○" for _ in range(correct)])
                else:
                    correct = 0
            if (episode+1) % self.params["log_freq"] == 0:
                self.show_log(episode, logs, mode)
            if (episode+1) % self.params["save_freq"] == 0:
                self.show_log(episode, logs, mode, save=True)
                save_model(self.agent.main_q_network, model_dir+str(mode)+'_DQN.pth')
            if correct >= self.params["patience"]:
                print("early stopping")
                save_model(self.agent.main_q_network, model_dir+str(mode)+'_DQN.pth')
                break
        
        finish_time = time.time()
        print("----------------Finish----------------", flush=True)
        print("Total Time (only Train):", datetime.timedelta(seconds=finish_time-start_time), flush=True)

    def train(self, mode="Train"):
 
        logs = {
            "episode_reward": [],
            "episode_loss_n": [], 
            "episode_loss_E": [],
            "episode_loss_l2": [],
            "episode_loss_all": [],
            "best_cost": [] 
        }
        best_count = 0

        total_step = 0  
        eps_random = self.params["eps_random"]
        eps_shift = self.params["eps_shift"]
      
        start_time = time.time()
        self.inter = start_time 
        print("----- "+str(mode)+" Start -----", flush=True)
        self.test(policy=["pred"])
        for episode in range(self.params["num_episodes"]):
            done = False
            good = False
            one_episode_transitions = []
            total_reward, total_loss_n, total_loss_E, total_loss_l2, total_loss_all = 0, 0, 0, 0, 0
            
            observation = self.env.reset(mode)
       
            state = torch.from_numpy(observation).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)
            
            action_list = []

            for step in range(self.params["max_steps"]):
                legal_actions = self.agent.get_legal_actions(state.squeeze())           
                action = self.agent.get_action(state, legal_actions, eps_random, eps_shift)
                action = torch.LongTensor([[action]])
                action_list.append(action.item())
                legal_actions = torch.from_numpy(legal_actions).type(torch.FloatTensor)

                action_ = self.agent.form_action(action)
                observation_next, reward, done = self.env.step(action_)

                if reward > 1:
                    good = True
                    best_count += 1
                    self.env.render(par_dir+'agent_partitions_episode'+str(episode+1)+'.png')
                    self.env.render(par_dir+'best_partitions.png')

                total_reward += reward
                reward = torch.FloatTensor([reward])
                
                if done:
                    state_next = None
                else:
                    state_next = torch.from_numpy(observation_next).type(torch.FloatTensor) 
                    state_next = torch.unsqueeze(state_next, 0)
                
                one_episode_transitions.append([state, action, state_next, reward, legal_actions])

                if total_step % self.params["update_main_freq"] == 0:
                    loss = self.agent.optimize_model(mode, episode, self.params["pre_num_episodes"])
                    if loss != None:
                        total_loss_n += loss[0].item()
                        total_loss_E += loss[1].item()
                        total_loss_l2 += loss[2].item()
                        total_loss_all += loss[3].item()
        
                if total_step % self.params["update_target_freq"] == 0:
                    hard_update_target_network(self.agent.main_q_network, self.agent.target_q_network)
            
                state = state_next
                total_step += 1

                if done:
                    print("selected action:{} reward:{}".format(action_list, reward.item()), flush=True)
                    for transition in one_episode_transitions:
                        if good:
                            self.agent.add_memory(transition[0], transition[1], transition[2], transition[3], True, transition[4])
                            self.demo_data.append([transition[0], transition[1], transition[2], transition[3], transition[4]])
                        else:
                            self.agent.add_memory(transition[0], transition[1], transition[2], transition[3], False, transition[4])
                    break
            
            logs["episode_reward"].append(total_reward)
            logs["episode_loss_n"].append(total_loss_n)
            logs["episode_loss_E"].append(total_loss_E)
            logs["episode_loss_l2"].append(total_loss_l2)
            logs["episode_loss_all"].append(total_loss_all)
            logs["best_cost"].append(self.env.best_cost)

            if (episode+1) % self.params["log_freq"] == 0:
                print("best update count: {}, demo memory size: {}".format(best_count, len(self.agent.demo_memory)), flush=True)
                self.show_log(episode, logs, mode)
            if (episode+1) % self.params["save_freq"] == 0:
                self.show_log(episode, logs, mode, save=True)
                save_model(self.agent.main_q_network, model_dir+str(mode)+'_episode'+str(episode+1)+'.pth')
                save_memory(self.demo_data, log_dir+'demo_agent_data.pickle')
        
        finish_time = time.time()
        print("----------------Finish----------------", flush=True)
        print("Total Time:", datetime.timedelta(seconds=finish_time-start_time), flush=True)
        print("Run Query Time:", datetime.timedelta(seconds=self.env.all_query_runtime), flush=True)
    
    def test(self, mode="Test", policy=["demo", "pred"]):
       
        for p in policy:
            action_list = []    
            done = False

            observation = self.env.reset(mode)
            state = torch.from_numpy(observation).type(torch.FloatTensor) 
            state = torch.unsqueeze(state, 0)
            
            for step in range(self.params["max_steps"]):
                if p == "pred":
                    legal_actions = self.agent.get_legal_actions(state.squeeze(0))
                    action = self.agent.get_action(state, legal_actions, test=True) 
                else:
                    action = self.env.get_demo_action()
                    if action is None:
                        print("Can't divide partitions by demo action．Recommend add more grids")
                        break
                action = torch.Tensor([[action]]).to(torch.int)
                action_list.append(action.item())

                action_ = self.agent.form_action(action)
                observation_next, _, done = self.env.step(action_)
  
                if done:
                    state_next = None
                else:
                    state_next = torch.from_numpy(observation_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)
                
                state = state_next

                if done:
                    print("{} action:{}".format(p, action_list), flush=True)
                    if p == "demo":
                        self.demo_action_list = action_list
                        self.env.render(par_dir+str(p)+'_partitions.png')
                    break

        if mode == "PreTrain":         
            if self.demo_action_list == action_list:
                return True
            else:
                return False
     
    def show_log(self, episode, logs, mode, save=False):
        if save:
            df_logs = pd.DataFrame(logs.values(), index=logs.keys()).T
            df_logs.to_csv(log_dir+'logs.csv')

            window = 1
            x = list(np.arange(len(df_logs)))
            r = df_logs["episode_reward"].rolling(window).mean()
            l_n = df_logs["episode_loss_n"].rolling(window).mean()
            l_E = df_logs["episode_loss_E"].rolling(window).mean()
            l_l2 = df_logs["episode_loss_l2"].rolling(window).mean()
            l_all = df_logs["episode_loss_all"].rolling(window).mean()
            best_cost = df_logs["best_cost"].rolling(window).mean()

            fig = plt.figure(figsize=(15,10))
            plt.plot(x, r, label="reward")
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            fig.savefig(log_dir+str(mode)+'_reward_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            fig = plt.figure(figsize=(15,10))
            plt.plot(x, l_n, label="loss_n")
            plt.plot(x, l_E, label="loss_E")
            plt.plot(x, l_l2, label="loss_l2")
            plt.plot(x, l_all,  label="loss_all")
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.legend()
            fig.savefig(log_dir+str(mode)+'_loss_history.png', dpi=300, bbox_inches='tight')
            plt.close()

            fig = plt.figure(figsize=(15,10))
            plt.plot(x, best_cost,  label="best_runtime")
            plt.xlabel('Episode')
            plt.ylabel('Query Runtime [sec]')
            plt.legend()
            fig.savefig(log_dir+str(mode)+'_best_runtime.png', dpi=300, bbox_inches='tight')
            plt.close()

        else:
            now = time.time()
            elapsed_time = now - self.inter
            print("epoch \t reward \t best_cost \t loss_all \t elapsed_time[sec]", flush=True)
            print("{:5d} \t {:.5f} \t {:.5f} \t {:.5f} \t {:.5f}".format(episode+1, logs["episode_reward"][episode], logs["best_cost"][episode], logs["episode_loss_all"][episode], elapsed_time), flush=True) # ログ出力
            self.inter = now
        
if __name__ == '__main__':

    log_dir = './logs/'+str(datetime.date.today())+'/'
    par_dir = log_dir+'partitions/'
    model_dir = log_dir+'models/'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(par_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open('./python/config.yaml', 'r', encoding="utf-8") as yml:
        config = yaml.safe_load(yml)

    SDP = Environment(config)
    SDP.make_demo()
    SDP.pretrain()
    SDP.train()