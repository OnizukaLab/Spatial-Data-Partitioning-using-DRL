import numpy as np
import torch
import pickle
import torch.nn as nn

def update_params(optim, loss, retain_graph=True):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

def hard_update_target_network(main, target):
    target.load_state_dict(main.state_dict())

def soft_update_target_network(tau, main, target):
    for target_param, main_param in zip(target.parameters(), main.parameters()):
        target_param.data.copy_(tau*main_param.data + (1.0-tau)*target_param.data)   

def copy_model(from_model, to_model):
    """Copies model parameters from from_model to to_model"""
    for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
        to_model.data.copy_(from_model.data.clone())

def save_model(model, path):
    torch.save(model.state_dict(), path)

def read_model(model, path):
    model.load_state_dict(torch.load(path))

def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False

def save_memory(memory, path):
    with open(path, 'wb') as f:
        pickle.dump(memory, f)

def read_memory(path):
    with open(path, 'rb') as f: 
        memory = pickle.load(f)
    return memory