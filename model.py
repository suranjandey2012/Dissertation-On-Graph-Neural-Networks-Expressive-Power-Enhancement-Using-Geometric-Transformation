
import torch
import torch.nn as nn
import torch.nn.functional as F


#Custom GNN Class that rotates the features and performs angular aggregation followed by classification 
class Custom_GNN(nn.Module):
    def __init__(self,L,input_dim,hidden_dim,out_dim,dropout,device):
        super(Custom_GNN,self).__init__()
        self.device=device #Set the device
        self.L=L #Defines the number of layers
        self.input_dim=input_dim #Define the input dimension
        self.hidden_dim=hidden_dim #Define the hidden dimension
        self.output_dim=out_dim #Define the final output dimension
        self.dropout=dropout   #Define the dropout
        self.Linear=nn.Linear(self.input_dim, self.hidden_dim,device=self.device) #Learnable Weight Matrix
        #stdv = 1. / math.sqrt(self.Linear.weight.size(1))
        #self.Linear.weight.data.uniform_(-stdv, stdv)
        #self.Linear.bias.data.uniform_(-stdv,stdv)
        torch.nn.init.xavier_uniform_(self.Linear.weight).to(self.device)
        torch.nn.init.zeros_(self.Linear.bias).to(self.device) #Bias initialization
        #torch.nn.init.xavier_uniform_(self.Linear.bias)
        self.classifier=nn.Sequential(
                                       #nn.Linear(self.hidden_dim,self.hidden_dim),
                                       nn.Linear(self.hidden_dim,self.hidden_dim,device=self.device),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(self.hidden_dim,device=self.device),
                                       nn.Linear(self.hidden_dim, self.output_dim,device=self.device)
                                       )
        self.layernorm=nn.LayerNorm(normalized_shape=(self.hidden_dim,),device=self.device)
        
    #Method that is used to compute the angular distance of nodes features from other nodes
    def Compute_And_Agg_Angular_Distance(self,features,A):        
        # Compute norms and normalize features
        norm = torch.norm(features, p=2, dim=1, keepdim=True)  # (n, 1)
        norm_inv = 1 / (norm + 1e-4)  # Avoid division by zero
        
        # Normalize feature vectors directly (element-wise)
        features_normalized = features * norm_inv  # (n, d)
        # Compute cosine similarity matrix directly
        Angle = torch.clamp(torch.matmul(features_normalized, features_normalized.T), -1, 1)

        # Normalize the adjacency matrix using sparse operations
        Degree = torch.sparse.sum(A, dim=1).to_dense().to(self.device)  # Shape: (num_nodes,)
        D_inv = torch.pow(Degree, -0.5)
        D_inv[torch.isinf(D_inv)] = 0  # Handle zero-degree nodes
        
        # Sparse Degree Normalization Matrix (Avoid dense diag)
        indices = torch.arange(A.shape[0], device=self.device).repeat(2, 1)
        D_inv_mat = torch.sparse_coo_tensor(indices, D_inv, (A.shape[0], A.shape[0])).to(self.device)
 
        # Efficient Sparse Normalization: D⁻¹ * A * D⁻¹
        A_norm = torch.sparse.mm(D_inv_mat, A)
        A_norm = torch.sparse.mm(A_norm, D_inv_mat)
 
        # Efficient Aggregation: A_norm * Angle
        support = torch.sparse.mm(A_norm, Angle.t())  # Shape: (n, n)
 
        # Extract diagonal efficiently
        updated_angle = torch.diagonal(support, offset=0).to(self.device)
 
        return updated_angle
    
    
    #Anti-Clockwise Rotation
    def rotate_features_with_matrix(self,features, angles, axis1=0, axis2=1):
        n, d = features.shape
        if not (0 <= axis1 < d and 0 <= axis2 < d):
            raise ValueError("Axes must be valid dimensions within the feature vector.")
    
        # Compute cos and sin for each node
        cos_theta = torch.cos(angles).view(-1, 1, 1)  # Shape: (n, 1, 1)
        sin_theta = torch.sin(angles).view(-1, 1, 1)  # Shape: (n, 1, 1)
        
        # Construct batched rotation matrices (n, 2, 2)
        rotation_matrix = torch.zeros((n, 2, 2), device=self.device)
        rotation_matrix[:, 0, 0] = cos_theta.squeeze()
        rotation_matrix[:, 1, 1] = cos_theta.squeeze()
        rotation_matrix[:, 0, 1] = -sin_theta.squeeze()
        rotation_matrix[:, 1, 0] = sin_theta.squeeze()
    
        # Extract and Rotate the first 2 dimensions
        rotated_part = torch.bmm(rotation_matrix, features[:, :2].unsqueeze(-1)).squeeze(-1)
        
        # Efficiently replace first two dimensions
        rotated_features = features.clone()
        rotated_features[:, :2] = rotated_part
    
        return rotated_features


    #Override The Forward Method
    def forward(self,input,A):
        #print(f"Before Applying Linear Layer:{input.isnan().any()}")
        #print(f"Minimum and the maximum values in the input:{input.min(),input.max()}")
        H=self.Linear(input).to(self.device) #Apply the linear layer on the features
        #print(f"After computing H:{H.isnan().any()}")
        #if(self.Linear.weight.isnan().any()):             #Debugging Nan's in linear layer weights
        #    print(self.Linear.weight)
        support=F.relu(H)    #Apply the relu activation
        #print(f"After computing support:{support.isnan().any()}")
        support1=self.layernorm(support).to(self.device) #Apply the layer norm on the features
        #print(f"After computing support1:{support1.isnan().any()}")
        input=F.dropout(support1,self.dropout,training=self.training).to(device=self.device) #Applying dropout on the features
        for i in range(self.L):
        #    print(f"Before Angle Computation in iteration{i+1}:{input.isnan().any()}")
            AGG_Angle=self.Compute_And_Agg_Angular_Distance(input,A) #Matrix that contains the angle made by the features of a node with respect to other nodes including iteself

            H_new=self.rotate_features_with_matrix(input, AGG_Angle, axis1=0, axis2=1).to(self.device)  #Rotate the feature vectors of the nodes using the aggregated angles
 
            input=H_new
        
        H_out=self.classifier(H_new.to(self.device)) #Pass the rotated features through the classifier
        return F.log_softmax(H_out,dim=1)
