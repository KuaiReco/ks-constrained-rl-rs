import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action

    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.sigmoid(self.l3(a))

class SL(object):
    def __init__(self, args):
        state_dim, action_dim, max_action=args.state_dim, args.action_dim, args.max_action
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.max_action=max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        #return self.actor(state).cpu().data.numpy().flatten()
        return self.actor(state)


    def train(self, replay_buffer, batch_size=64):
        # Sample replay buffer 
        state, action, _,_,_= replay_buffer.sample(batch_size)

        predict_action=self.actor(state)
        action=action.clamp(-self.max_action, self.max_action)
        #print("predict_action: ", predict_action)
        #print("action: ", action)
        loss=F.mse_loss(predict_action,action)

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location="cpu"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location="cpu"))
    
        