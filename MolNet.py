import torch
import torch.nn as nn

import json
from collections import defaultdict
#import matplotlib.pyplot as plt
import os
import glob






class MolNet(nn.Module):
    def __init__(self, input_nodes=1,hidden_nodes=1,
                 output_nodes=1,layers=1, activator=None, loss=None, dev="cpu",
                 with_dropouts = False, only_last_dropout = False, dropout_rate=0.5):
        def active_fail():
            print(F"Fail! The functions {list(self.chooser.keys())} are supported")
            return "Not a supported function"
        def loss_fail():
            print(F"Fail! The loss functions supported are {list(self.losses.keys())}")
            return "Not a supported function"
        self.losses = defaultdict(loss_fail)
        self.losses["mse"] = nn.MSELoss
        self.losses["mae"] = nn.L1Loss
        self.chooser = defaultdict(active_fail)
        self.chooser["sigmoid"] = nn.Sigmoid()
        self.chooser["relu"] = nn.ReLU()
        self.chooser["softmax"] = nn.Softmax()
        super(MolNet, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(layers):

            if i == 0:
                self.hidden_layers.append(nn.Linear(input_nodes,hidden_nodes))

                if with_dropouts:
                    self.hidden_layers.append(nn.Dropout(p=dropout_rate))
            if i != 0:
                self.hidden_layers.append(nn.Linear(hidden_nodes,hidden_nodes))
            if with_dropouts and i !=0:
                self.hidden_layers.append(nn.Dropout(p=dropout_rate))

        if only_last_dropout:
            self.hidden_layers.append(nn.Dropout(p = dropout_rate))
        self.output = nn.Linear(hidden_nodes, output_nodes)
        for layer in self.hidden_layers:
            if isinstance(layer, torch.nn.modules.dropout.Dropout):
                continue
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(layer.bias)

        self.activation_function = self.chooser[activator]
        self.loss_function = self.losses[loss]
        self.my_device = dev

    def forward(self, features):
        features = features.to(self.my_device)


        features = torch.flatten(features, 1)
        for layer in range(len(self.hidden_layers)):
            features=self.hidden_layers[layer](features)
            features = self.activation_function(features)

        out = self.output(features)
        return out
    

    
    
    
    
    
    
    
    
    
    
