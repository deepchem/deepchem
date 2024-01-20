from typing import Iterable, List, Tuple
from deepchem.data import Dataset
import torch
import numpy as np
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.torch_models import layers
import torch.nn as nn


default_dict = {
    '#': 1,
    '(': 2,
    ')': 3,
    '+': 4,
    '-': 5,
    '/': 6,
    '1': 7,
    '2': 8,
    '3': 9,
    '4': 10,
    '5': 11,
    '6': 12,
    '7': 13,
    '8': 14,
    '=': 15,
    'C': 16,
    'F': 17,
    'H': 18,
    'I': 19,
    'N': 20,
    'O': 21,
    'P': 22,
    'S': 23,
    '[': 24,
    '\\': 25,
    ']': 26,
    '_': 27,
    'c': 28,
    'Cl': 29,
    'Br': 30,
    'n': 31,
    'o': 32,
    's': 33
}


class TextCNN(TorchModel):

    def __init__(self,
                 n_tasks,
                 char_dict,
                 seq_length,
                 n_embedding=75,
                 kernel_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                 num_filters=[
                     100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160
                 ],
                 dropout=0.25,
                 mode="classification",
                 **kwargs):
        
        self.n_tasks = n_tasks
        self.char_dict = char_dict
        self.seq_length = max(seq_length, max(kernel_sizes))
        self.n_embedding = n_embedding
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout = dropout
        self.mode = mode

        self.conv_layers = nn.ModuleList()
        self.embedding_layer = layers.DTNNEmbedding(
            n_embedding=self.n_embedding,
            periodic_table_length=len(self.char_dict.keys()) + 1)
        self.dropout_layer = nn.Dropout1d(p = self.dropout)
        self.linear = nn.Linear(in_features=,out_features=200)
        self.relu = nn.ReLU()
        
        for filter_size, num_filter in zip(self.kernel_sizes, self.num_filters):
            self.conv_layers.append(nn.Conv1d(in_channels=self.n_embedding,
                                              out_channels=num_filter,
                                              kernel_size=filter_size
                                              ))

    def forward(self, input):

        input_emb = self.embedding_layer(input)

        for conv_layer in self.conv_layers:
            x = conv_layer(input_emb)
            x,_ = torch.max(x,dim=2)
            concat_output = torch.cat(x, dim=1)
        
        x = self.relu(self.linear(concat_output))
        
        

        
        








    
    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
    
