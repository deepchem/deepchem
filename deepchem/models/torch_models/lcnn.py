import torch
import torch.nn as nn
import numpy as np


class LCNNConvolution(nn.Module):
    def __init__(self ,input_feature , output_feature  , dropout = None,UseBN = False  ):
        
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
        
    def forward(self , X_sites , X_NSs , N_sites ):
        """
        X_Sites: 2D tensor --> (Number of sites , size of node_feature)
        
        X_NSs: 3D tensor --> (Number of sites , Number of permutation , neighbor size)
        
        
        """
        for i in range(0 , len(X_NSs)):
            
            # Neighbors of the current node i along with all the permutations 
            X_NSs_i = X_NSs[i]
            # Number of permutation * (node_feature_size * node_size) 
            X_site_i = X_sites[X_NSs_i[i]].view( X_NSs_i[i].shape[0],-1 )            
            # Number of permutation * depth
            X_1 = self.conv_weights(X_site_i)

            