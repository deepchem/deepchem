import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LCNNConvolution(nn.Module):
    def __init__(self ,input_feature , output_feature  , dropout = 0.2,UseBN = False  ):
        
        """
        Lattice Convolution Layer used in the main model
        
        parameters
        -----------
        
        input_feature: Dimenion of the concatenated input vector. Node_feature_size*number of neighbors
        output_feature / depth: Dimension of feature size of the convolution
        dropout: Dropout
        UseBN: To use batch normalisation
        
        """
        super(LCNNConvolution , self).__init__()
        
        self.conv_weights = nn.Linear(input_feature , output_feature)
        self.batch_norm = nn.BatchNorm1d(output_feature)
        self.UseBN = UseBN
        self.activation = Shifted_softplus()
        self.dropout = Custom_dropout(dropout)
        
        
    def forward(self , X_sites , X_NSs  ):
        """
        X_Sites: 2D tensor --> (Number of sites , size of node_feature)
        
        X_NSs: 3D tensor --> (Number of sites , Number of permutation , neighbor size)
        
        
        """
        updates = []
        for i in range(0 , len(X_NSs)):
            
            # Neighbors of the current node i along with all the permutations

            X_NSs_i = X_NSs[i]
            # Number of permutation * (node_feature_size * node_size) 
            X_site_i = X_sites[[X_NSs_i]].view( X_NSs_i.shape[0],-1 ) .float()
            # Number of permutation * depth
            X_1 = self.conv_weights(X_site_i)
            
            if self.UseBN:
                X_1 = self.batch_norm(X_1)
            
            # Shifted Softplus activatiion function
            X_2 = self.activation(X_1)
            # It zeros out few rows few permutations
            X_3 = self.dropout(X_2)
            X_4 = X_3.sum(axis = 0)
            updates.append(X_4)
        
        updates = torch.stack(updates , axis = 0)
        
        return updates
            
            
                
class Shifted_softplus(nn.Module):
    def __init__(self ):
        
        super(Shifted_softplus , self).__init__()
        self.act = nn.Softplus()
    
    def forward(self , X):
        
        return self.act(X) - torch.log(torch.tensor([2.00]))
    

                
class Custom_dropout(nn.Module):
    def __init__(self , dp_rate):
        super(Custom_dropout , self).__init__()
        
        self.m = nn.Dropout(p=dp_rate)
    def forward(self , layer):
        temp = torch.ones(layer.shape[0])
        cust = self.m(temp).view(layer.shape[0] , 1).repeat(1,layer.shape[1])
        return cust*layer
     