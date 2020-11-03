import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LCNNBlock(nn.Module):
    """
    The Lattice Convolution layer of LCNN
    
    The following class implements the lattice convolution function which is
    based on graph convolution networks where,
    [1] Each atom is represented as a node
    [2] Adjacent atom based on distance are considered as neighbors.
    
        
        
    Operations in Lattice Convolution:
    
    [1] In graph aggregation step- node features of neighbors are concatenated and
        into a linear layer. But since diffrent permutation of order of neighbors could
        be considered because of which , diffrent permutation of the lattice 
        structure are considered in diffrent symmetrical angles (0 , 60 ,120 180 , 240 , 300 )
        
    [2] Diffrent permutations are added up for each node and each node is transformed
        into a vector.
    """
    def __init__(self ,input_feature , output_feature = 44 , dropout = 0.2,UseBN = True  ):
        
        """
        Lattice Convolution Layer used in the main model
        
        parameters
        -----------
        
        input_feature: Dimenion of the concatenated input vector. Node_feature_size*number of neighbors
        output_feature / depth: Dimension of feature size of the convolution
        dropout: Dropout
        UseBN: To use batch normalisation
        
        """
        super(LCNNBlock , self).__init__()
        
        self.conv_weights = nn.Linear(input_feature , output_feature)        
        self.bias = nn.Parameter(torch.Tensor(1, output_feature))     
        self.batch_norm = nn.BatchNorm1d(output_feature)
        self.UseBN = UseBN
        self.activation = Shifted_softplus()
        self.dropout = Custom_dropout(dropout)
        
        
    def forward(self , feed_dict   ):
        """
        X_Sites: 2D tensor --> (Number of sites , size of node_feature)
        
        X_NSs: 3D tensor --> (Number of sites , Number of permutation , neighbor size)
        
        feed_dict["X_Sites"] , feed_dict["X_NSs"]        
        
        """
        updates = []
        for i in range(0 , len(feed_dict["X_NSs"])):
            
            # Neighbors of the current node i along with all the permutations

            X_NSs_i = feed_dict["X_NSs"][i]
            # Number of permutation * (node_feature_size * node_size) 
            X_site_i = feed_dict["X_Sites"][[X_NSs_i]].view( X_NSs_i.shape[0],-1 ) .float()
            # Number of permutation * depth
            X_1 = self.conv_weights(X_site_i) + self.bias
            if self.UseBN:
                X_1 = self.batch_norm(X_1)
            
            # Shifted Softplus activatiion function
            X_2 = self.activation(X_1)
            # It zeros out few rows few permutations
            X_3 = self.dropout(X_2)
            X_4 = X_3.sum(axis = 0)
            updates.append(X_4)
        
        updates = torch.stack(updates , axis = 0)
        
        return {"X_Sites":updates , "X_NSs":feed_dict["X_NSs"]}  

class Atom_Wise_Linear(nn.Module):
    """
    Performs Matrix Multiplication
    
    It is used to transform each node wise feature into a scalar
    
    """
    def __init__(self , input_feature , output_feature , dropout = 0.0 ,UseBN = True):
        """
        input_feature: Size of input feature size
        output_feature: Size of output feature size
        
        """
        super(Atom_Wise_Linear , self).__init__()
        self.conv_weights = nn.Linear(input_feature , output_feature)
        self.bias = nn.Parameter(torch.Tensor(1, output_feature))
    def forward(self ,X_sites ):
        X_1 = self.conv_weights(X_sites) + self.bias
        return X_1
        
        
        
class Atom_Wise_Convolution(nn.Module):
    """
    Performs self convolution to each node
    """
    def __init__(self , input_feature , output_feature , dropout = 0.2 ,UseBN = True):
        """
        input_feature: Size of input feature size
        output_feature: Size of output feature size
        """
        super(Atom_Wise_Convolution , self).__init__()
        self.conv_weights = nn.Linear(input_feature , output_feature)
        self.bias = nn.Parameter(torch.Tensor(1, output_feature))        
        self.batch_norm = nn.BatchNorm1d(output_feature)
        self.UseBN = UseBN
        self.activation = Shifted_softplus()
        self.dropout = Custom_dropout(dropout)
        
    def forward(self ,X_sites ):
                
        X_1 = self.conv_weights(X_sites) + self.bias
        if self.UseBN:
            X_1 = self.batch_norm(X_1)
            
        X_2 = self.activation(X_1)
        X_3 = self.dropout(X_2)
        
        return X_3          
        
                
class Shifted_softplus(nn.Module):
    
    """
    This code for Activation Function .
    """
    def __init__(self ):        
        super(Shifted_softplus , self).__init__()
        self.act = nn.Softplus()
    
    def forward(self , X):        
        return self.act(X) - torch.log(torch.tensor([2.00]))
    

                
class Custom_dropout(nn.Module):
    """
    An implementation for few , Given a task perform a rowise sum of 2-d
    matrix , you get a zero out the contribution of few of rows in the matrix

    Given, X a 2-d matrix consisting of row vectors (1-d) x1 , x2 ,..xn.
    Sum = x1 + 0.x2 + .. + 0.xi + .. +xn
    """
    def __init__(self , dp_rate):
        super(Custom_dropout , self).__init__()
        
        self.m = nn.Dropout(p=dp_rate)
    def forward(self , layer):
        temp = torch.ones(layer.shape[0])
        cust = self.m(temp).view(layer.shape[0] , 1).repeat(1,layer.shape[1])
        return cust*layer

    
class LCNNModel(nn.Module):
    """
    The Lattice Convolution Neural Network (LCNN)
    
    This model takes lattice representation of Adsorbate Surface to predict
    coverage effects taking into consideration the adjacent elements interaction
    energies.
    
    The model follows the following steps
    
    [1] It performs n lattice convolution operations. 
        For more details look at the LCNNBlock class
    [2] Followed by Linear layer transforming into sitewise_n_feature
    [3] Transformation to scalar value for each node.
    [4] Average of properties per each element in a configuration
    
    Refrences
    -----------
    
    [1] Jonathan Lym,Geun Ho Gu, Yousung Jung , and Dionisios G. Vlachos 
    "Lattice Convolutional Neural Network Modeling of Adsorbate Coverage 
    Effects" The Journal of Physical Chemistry
    
    [2] https://forum.deepchem.io/t/lattice-convolutional-neural-network-modeling-of-adsorbate-coverage-effects/124
    
    Notes
    -----
    This class requires PyTorch to be installed.

    """
    def __init__(self,
                 n_occupancy,
                 n_neighbor_sites_list,
                 n_permutation_list,
                 dropout_rate = 0.2,
                 n_conv = 2,
                 n_features  = 150,
                 sitewise_n_feature = 25):
        """
        parameters
        ----------
        
        n_occupancy: int. number of possible occupancy
        n_neighbor_sites_list: list of int. Number of neighbors of each site. 
        n_permutation_list: Diffrent permutations taken along diffrent directions.
        nconv: int. number of convolutions performed
        n_feature: int. number of feature for each site
        sitewise_n_feature: int. number of features for atoms for site-wise activation
        
        """
        super(LCNNModel , self).__init__()
        
        modules = [LCNNBlock(n_occupancy*n_neighbor_sites_list , n_features  )]
        for i in range(n_conv-1):
            modules.append(LCNNBlock(n_features*n_neighbor_sites_list, n_features ))
            
        
        self.LCNN_blocks = nn.Sequential(*modules)
        self.Atom_wise_Conv = Atom_Wise_Convolution(n_features , sitewise_n_feature)
        self.Atom_wise_Lin = Atom_Wise_Linear(sitewise_n_feature , sitewise_n_feature)
        
        
        
    def forward(self,X_sites , X_NSs , N_Sites_per_config ,Idx_Config ):
        
        X_1 = self.LCNN_blocks({"X_Sites":X_sites , "X_NSs":X_NSs})["X_Sites"]
        X_2 = self.Atom_wise_Conv(X_1)
        X_3 = self.Atom_wise_Lin(X_2).sum(axis = 1)
        X_4 = torch.zeros(N_Sites_per_config.shape[0]).scatter_add_(0, torch.tensor( dic['Idx_Config']).view(-1),X_3 )
        X_5 = X_4 / N_Sites_per_config  
        
        return  X_5
        
        
        
        
        