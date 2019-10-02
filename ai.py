# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch #supports Dynamic graphs
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim #optimize Stochastic Gradient Descent
import torch.autograd as autograd
from torch.autograd import Variable #to convert tensor to a vriable witch also contains the gradient

# Creating the architecture of the Neural Network

class Network(nn.Module): # torch.nn.Module is a Base class for all neural network modules
    #input_size - number of input neurons in the neural network
    #nb_action - number of output neurons in the neural networ
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        #Linear applies a linear transformation to the incoming data
        #for full connection between input layer and the hidden layer:
        # first arg - number of  neurons in the first layer (input layer in our case)
        #second arg - number of neurons in the second layer (hidden layer in our case), by experience we set it to 30
        #third arg - include bias (true by default)
        self.fc1 = nn.Linear(input_size, 30)
        #for full connection between hidden layer and the output layer:
        # first arg - number of  neurons in the first layer (hidden layer in our case)
        #second arg - number of neurons in the second layer (output layer in our case)
        #third arg - include bias (true by default)
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state):
        #x represent the hidden layer
        #relu applies the rectified linear unit function element-wise , in our case on the hidden layer
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    #pushing events to our memory
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    #sample batch from memory
    def sample(self, batch_size):
        #rearrange the events of the random sample from the memory to groups, 
        #for instance all the states in one group, all the actions in second group, etc.
        samples = zip(*random.sample(self.memory, batch_size))
        #torch.cat - Concatenates the given sequence of seq tensors in the given dimension.
        #We are converting x (the samples) once lambda is applied
        #but because each batch thats in the sample (a1,a2, a3, etc)
        #we have to concatenate it to the first dimension corresponding to the state so that it is aligned
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        #sliding window of the last rewards, 1000 in our cas
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        #Adam class: for an algorithm for first-order gradient-based optimization
        #of stochastic objective functions, based on adaptive estimates of lower-order moments
        #first arg - iterable of parameters to optimize or dicts defining parameter groups
        #second arg - learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        #unsqueeze -  returns a new tensor with a dimension of size one inserted at the specified position
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0 # in our case will hold 0, 1 or 2
        self.last_reward = 0 # in our case will hold number between -1 to 1
    
    def select_action(self, state):
        #temperature parameters T , we are using it to basically control the "certainty" of
        #the decisions by the AI. For example the closer to 0 the less certain and
        #the higher you increase it the more certain that the AI's decisions will be.
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        #chosen action
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #self.model(batch state) takes all the input states in your batch, 
        #and predicts the Q-values for all of them. 
        #The network object is calling the forward function to return the predictions.
        #gather - to get only the action actualy choosen
        #unsqueeze1 - The batch state has a "fake" dimension and batch action doesn't have it.
        #We have to unsqueeze the actions so that the batch action has the same dimension as the batch state.
        #We use 0 for the fake dimension of the state and 1 for the actions.
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0] # 1 means we want max according to the action
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target) # temporal difference loss, Huber loss in our case
        #zero_grad - reinitialize the optimizer in each iteration of the loop
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True) #back propagation of error
        self.optimizer.step() #updates the weights
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    #return the mean of rewards in the sliding window
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    #state_dict() - returns a dictionary containing a whole state of the module.
    # last param of the save func: the name of the saved file
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")