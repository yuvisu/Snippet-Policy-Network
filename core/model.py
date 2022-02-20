import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from core.layer import BaseCNN, BaseRNN, Controller
from core.layer import BaselineNetwork, Discriminator
from core.loss import FocalLoss

class SnippetPolicyNetwork(nn.Module):

    def __init__(self, input_size = 12, hidden_size = 256, hidden_output_size = 1, output_size = 9):
        super(SnippetPolicyNetwork, self).__init__()
        self.loss_func = FocalLoss() # It is the normal cross-entropy function, as the gamma is set to zero
        self.CELL_TYPE = "LSTM"
        self.INPUT_SIZE = input_size
        self.HIDDEN_SIZE = hidden_size
        self.HIDDEN_OUTPUT_SIZE = hidden_output_size
        self.OUTPUT_SIZE = output_size
        
        self._ALPHA = 0.01
        self._EPISOLON = 1
        self._LIMIT = 10
        
        # --- Backbone Networks ---#        
        self.BaseCNN = BaseCNN(input_size, hidden_size, output_size).cuda()
        self.BaseRNN = BaseRNN(hidden_size, hidden_size, self.CELL_TYPE).cuda()
        
        # --- Controlling Agent ---#  
        self.Controller = Controller(hidden_size, hidden_output_size).cuda()
        self.BaselineNetwork = BaselineNetwork(hidden_size, hidden_output_size).cuda()
        
        # --- Discriminator ---#  
        self.Discriminator = Discriminator(hidden_size, output_size).cuda()
            
    def initHidden(self, batch_size, weight_size):
        """Initialize hidden states"""
        if self.CELL_TYPE == "LSTM":
            h = (torch.zeros(1, batch_size, weight_size).cuda(),
                 torch.zeros(1, batch_size, weight_size).cuda())
        else:
            h = torch.zeros(1, batch_size, weight_size).cuda()
        return h

    def forward(self, X):
        
        hidden = self.initHidden(len(X),self.HIDDEN_SIZE)
        
        '''
            due to the varied length of snippet array in batch training
            we need to find the max_length in each batch
            then maintain a list to record the stopping point of each sample in the same batch
        '''
        min_length = 1000
        max_length = 0
        
        for x in X:
            if min_length > x.shape[0]:
                min_length = x.shape[0]
            if max_length < x.shape[0]:
                max_length = x.shape[0]
        
        tau_list = np.zeros(X.shape[0], dtype=int) # maintain stopping point for each sample in a batch
        state_list = np.zeros(X.shape[0], dtype=int) # record which snippet array already stopped in a batch
        
        log_pi = []
        baselines = []
        halt_probs = []
        
        flag = False
                
        for t in range(max_length):
            
            slice_input = []
            cnn_input = None
               
            '''
            In the training phase, we need to find the next snippet according to the maintaining list
            as different snippet array stops in different time point in the same batch
            
            In the testing phase, use the time index directly
            '''
            if self.training:        
                for idx, x in enumerate(X):
                    slice_input.append(torch.from_numpy(x[tau_list[idx],:,:]).float())
                    cnn_input = torch.stack(slice_input, dim=0)
            else:
                for idx, x in enumerate(X):
                    slice_input.append(torch.from_numpy(x[t,:,:]).float())
                    cnn_input = torch.stack(slice_input, dim=0)

            cnn_input = cnn_input.cuda()

            # --- Backbone Network ---
            S_t = self.BaseCNN(cnn_input).unsqueeze(0)
                        
            S_t, hidden = self.BaseRNN(S_t,hidden)
            
            H_t = hidden[0][-1]

            rate = self._LIMIT if t > self._LIMIT else t
            
            # --- Controlling Agent ---
            if self.training:
                #set different alpha can affect the performance
                a_t, p_t, w_t, probs = self.Controller(H_t, self._ALPHA, self._EPISOLON) # Calculate if it needs to output
            else:
                a_t, p_t, w_t, probs = self.Controller(H_t, self._ALPHA * rate, self._EPISOLON) # Calculate if it needs to output
            
            # --- Baseline Network ---
            b_t = self.BaselineNetwork(H_t)
                        
            # --- Discriminator ---
            y_hat = self.Discriminator(H_t)
            
            log_pi.append(p_t)
            
            halt_probs.append(w_t)
            
            baselines.append(b_t)
            
            if self.training:
                for idx, a in enumerate(a_t):
                    
                    if(a == 0 and tau_list[idx] < X[idx].shape[0]-1):
                        tau_list[idx]+=1
                    else: 
                        state_list[idx] = 1
                        #record the stopping status of a snippet array in the batch
                        
                if (np.mean(state_list)>=1): break 
                # break condition in training phase
                # when all snippet array are stopped, the program will break
                    
            else:
                                
                for idx, a in enumerate(a_t):
                    
                    tau_list[idx] = t
                    
                    if(a == 1): flag = True
                        
                if(flag): break # break condition in testing phrase
                    
        self.log_pi = torch.stack(log_pi).transpose(1, 0).squeeze(2)
        
        self.halt_probs = torch.stack(halt_probs).transpose(1, 0)
        
        self.baselines = torch.stack(baselines).squeeze(2).transpose(1, 0)

        self.tau_list = tau_list
                
        return y_hat, tau_list

    def applyLoss(self, y_hat, labels, gamma = 1, beta = 0.001):
        # --- compute reward ---
        _, predicted = torch.max(y_hat, 1)
        r = (predicted.float().detach() == labels.float()).float() #checking if it is correct
        
        r = r*2 - 1 # return 1 if correct and -1 if incorrect
        
        R = torch.from_numpy(np.zeros((self.baselines.shape[0],self.baselines.shape[1]))).float().cuda()
        
        for idx in range(r.shape[0]):
            R[idx][:self.tau_list[idx]+1] = r[idx]
                
        # --- subtract baseline from reward ---
        adjusted_reward = R - self.baselines.detach()
        
        # --- compute losses ---
        self.loss_b = F.mse_loss(self.baselines, R) # Baseline should approximate mean reward
        self.loss_c = self.loss_func(y_hat, labels) # Make accurate predictions
        self.loss_r = torch.sum(-self.log_pi*adjusted_reward, dim=1).mean()
        self.time_penalty = torch.sum(self.halt_probs, dim=1).mean()
        
        # --- collect all loss terms ---
        loss = self.loss_c + gamma * self.loss_r + self.loss_b + beta * self.time_penalty
        
        return loss, self.loss_c, self.loss_r, self.loss_b, self.time_penalty
