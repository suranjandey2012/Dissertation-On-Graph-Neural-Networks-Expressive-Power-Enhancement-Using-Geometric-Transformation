import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import global_add_pool,global_mean_pool


#Custom GNN new variant that incorporates both the angular distance and spatial relationship defined using coordinates of n dim
class Custom_GNN(nn.Module):
    def __init__(self,L,input_dim,hidden_dim,out_dim,dropout,device,flag,readout='add',dataset_name='',act_fn=nn.SiLU(),emb_dim1=0,emb_dim2=0):
        super(Custom_GNN,self).__init__()
        self.device=device #Set the device
        self.readout=readout #Set the Graph Level Pooling function
        self.dataset_name=dataset_name #Set the dataset name
        self.L=L #Defines the number of layers
        self.input_dim=input_dim #Define the input dimension
        self.hidden_dim=hidden_dim #Define the hidden dimension
        self.output_dim=out_dim #Define the final output dimension
        self.dropout=dropout   #Define the dropout
        self.Linear=nn.Linear(self.input_dim,self.hidden_dim,device=self.device,dtype=torch.float32) #Learnable Weight Matrix
        self.message_weight=nn.Linear(self.hidden_dim,self.hidden_dim,device=self.device,dtype=torch.float32) #Linear layer to be applied on message weight
        self.embedding_h = nn.Embedding(emb_dim1,emb_dim2) #Embedding Layer
        self.w1=nn.Linear(2*self.hidden_dim,self.hidden_dim,device=self.device,dtype=torch.float32) #w1
        self.w2=nn.Linear(2,hidden_dim,device=self.device,dtype=torch.float32) #w2
        self.w3=nn.Linear(2,hidden_dim,device=self.device,dtype=torch.float32) #w3
        self.w_rotate = nn.Linear(self.hidden_dim,self.hidden_dim,device=self.device,dtype=torch.float32)
        self.layernorm=nn.LayerNorm(normalized_shape=(self.hidden_dim,),eps=1e-3,device=self.device,dtype=torch.float32)
        self.classifier=nn.Sequential(
                                       nn.Linear(self.hidden_dim,self.hidden_dim,device=self.device,dtype=torch.float32),
                                       nn.ReLU(),
                                       nn.BatchNorm1d(self.hidden_dim,device=self.device,dtype=torch.float32),
                                       nn.Linear(self.hidden_dim,self.output_dim,device=self.device,dtype=torch.float32)
                                       )
        self.flag=flag #Flag used to indicate whether classification or regression target
        self.node_dec = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim,device=self.device,dtype=torch.float32),
                                      act_fn,
                                      nn.Linear(self.hidden_dim, self.hidden_dim,device=self.device,dtype=torch.float32))  #MLP1 For Node wise feature expressivity
        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim,device=self.device,dtype=torch.float32),  
                                       act_fn,
                                       nn.Linear(self.hidden_dim, 1,device=self.device,dtype=torch.float32))        #MLP2 For Graph Level Prediction post pooling

        torch.nn.init.xavier_uniform_(self.w1.weight) #Xavier Initialization
        torch.nn.init.xavier_uniform_(self.w2.weight) #Xavier Initialization
        torch.nn.init.xavier_uniform_(self.w3.weight) #Xavier Initialization
        torch.nn.init.xavier_uniform_(self.Linear.weight) #Xavier Initialization
        torch.nn.init.xavier_uniform_(self.w_rotate.weight)
        torch.nn.init.zeros_(self.w1.bias) #Bias initialization
        torch.nn.init.zeros_(self.w2.bias) #Bias initialization
        torch.nn.init.zeros_(self.w3.bias) #Bias initialization
        torch.nn.init.zeros_(self.Linear.bias) #Bias initialization
        torch.nn.init.zeros_(self.w_rotate.bias)
        self.to(self.device)

    #Method that is used to compute angular distnace between 2 features and the respective coordinates     
    def Compute_Angular_Distance(self,features:torch.tensor,x:torch.tensor) -> tuple:
        # Compute L2 norms and prevent division by zero
        f_norm = torch.norm(features, p=2, dim=1, keepdim=True)  # (N, 1)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # (N, 1)
    

        # Avoid division by zero by replacing zeros with a small epsilon
        #f_norm = torch.where(f_norm == 0, torch.tensor(1e-8, device=f_norm.device), f_norm)
        #x_norm = torch.where(x_norm == 0, torch.tensor(1e-8, device=x_norm.device), x_norm)

        # Normalize feature and coordinate vectors
        features = features / (f_norm + 1e-4)  # Avoid division by zero
        x = x / (x_norm + 1e-4)  # Avoid division by zero
        
        # Compute pairwise cosine similarities
        features_angle = torch.matmul(features, features.T)  # (N, N)
        x_angle = torch.matmul(x, x.T)  # (N, N)
    
        # Ensure values are in the valid range [-1, 1] for arccos
        theta_f = torch.acos(torch.clamp(features_angle, -0.99, 0.99))
        theta_x = torch.acos(torch.clamp(x_angle, -0.99, 0.99))
        
        return theta_f, theta_x


    #Method that is used to compute the message vectors between 2 neighbours
    def Compute_Message_Vector(self,A:torch.tensor,features:torch.sparse.Tensor,theta_f:torch.tensor,theta_x:torch.tensor) -> torch.tensor:
        edges=A.coalesce().indices()  #First fetch the indices from the adjacency matrix
        
        H_i=features[edges[0]] #hi
        H_j=features[edges[1]] #hj

        Theta_h_ij=theta_f[edges[0],edges[1]]
        Theta_x_ij=theta_x[edges[0],edges[1]]

        M_h=torch.hstack((H_i,H_j))                                                                          #[h(i)||h(j)]
        M_theta_h=torch.hstack((torch.cos(Theta_h_ij.unsqueeze(-1)),torch.sin(Theta_h_ij.unsqueeze(-1))))    #[cos(theta_h(ij))||sin(theta_h(i,j))]
        M_theta_x=torch.hstack((torch.cos(Theta_x_ij.unsqueeze(-1)),torch.sin(Theta_x_ij.unsqueeze(-1))))    #[cos(theta_x(i,j))||sin(theta_x(i,j))]
         
        
        M=self.w1(M_h)+self.w2(M_theta_h)+self.w3(M_theta_x)

        message=F.relu(M)
        
        return message


    #Method that is used to update the coordinates of the individual vertices
    def Coordinate_Update(self,x:torch.tensor,M:torch.tensor,A:torch.sparse.Tensor) -> torch.tensor:
        row=A.coalesce().indices()[0] #Extract the rows

        column=A.coalesce().indices()[1] #Extract the columns

        x_i=x[row]  #Get the coordinates of the starting vertices of the edges

        x_j=x[column] #Get the coordinates of the ending vertices of the edges
        
        update=(x_i-x_j)*self.message_weight(M)  #(x_i-x_j)*M for all edges

        new=torch.zeros_like(x,dtype=torch.float32,device=self.device) #Initialize a Zero tensor
        
        x=x + new.scatter_add_(0,row.unsqueeze(1).expand(-1,x.shape[1]),update) #x_new=X+sum(x_i-x_j)*theta(m(i,j)) for all j neighbours
        
        return x
    

    #Method that is used to compute the aggregated angles for all nodes
    def Compute_Aggregated_Angle(self,x:torch.tensor,A:torch.sparse.Tensor):
        A = A.coalesce()  # Ensure COO format
        row, col = A.indices()  # Extract source (i) and target (j) nodes

        # Normalize node coordinates to unit vectors (avoid division issues)
        x = F.normalize(x, p=2, dim=1, eps=1e-6)  # (n, d)

        # Compute dot products for edges (i, j) -> x(i) . x(j)
        dot_products = (x[row] * x[col]).sum(dim=1)  # (num_edges,)

        # Clamp values to avoid NaNs in acos (ensure range [-1, 1])
        dot_products = torch.clamp(dot_products, -0.99, 0.99)

        # Compute angular distance (theta_ij = arccos(x(i) . x(j)))
        theta_ij = torch.acos(dot_products)  # (num_edges,)

        # Aggregate angles for each node using scatter_add
        aggregated_theta = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)  # (n,)
        aggregated_theta.scatter_add_(0, row, theta_ij)  # Sum θ_ij for each node i

        return aggregated_theta
    

    #Method used to apply a gated rotation to features
    def rotate_features_with_matrix2(self, features: torch.tensor, angles: torch.tensor) -> torch.tensor:
        n, d = features.shape
    
        # Compute cos and sin for each node
        cos_theta = torch.cos(angles).view(-1, 1, 1)  # Shape: (n, 1, 1)
        sin_theta = torch.sin(angles).view(-1, 1, 1)  # Shape: (n, 1, 1)
    
        # Placeholder for updated features
        updated_features = torch.zeros_like(features, dtype=torch.float32, device=self.device)
    
        # Iterate through all possible 2D subspaces using step of 2
        for i in range(0, d, 2):
            index_0, index_1 = i, i + 1
            
            # Check for dimension overflow (for odd dimensions)
            if index_1 >= d:
                break
            
            # Create a 2x2 rotation matrix for each node
            rotation_matrix = torch.zeros((n, 2, 2), device=self.device, dtype=torch.float32)
            rotation_matrix[:, 0, 0] = cos_theta.squeeze()
            rotation_matrix[:, 1, 1] = cos_theta.squeeze()
            rotation_matrix[:, 0, 1] = -sin_theta.squeeze()
            rotation_matrix[:, 1, 0] = sin_theta.squeeze()
    
            # Rotate the selected 2D subspace
            rotated_part = torch.bmm(rotation_matrix, features[:, i:i+2].unsqueeze(-1)).squeeze(-1)
    
            # Replace the rotated values
            rotated_features = features.clone()
            rotated_features[:, i:i+2] = rotated_part
    
            # Gating mechanism using a learnable function
            gate_weights = torch.sigmoid(self.w_rotate(rotated_features))
    
            # Aggregate using the gate
            updated_features += gate_weights * rotated_features
    
        return updated_features

    #Method that is used to perform Anti-Clockwise Rotation of the feature vectors
    def rotate_features_with_matrix(self,features:torch.tensor, angles:torch.tensor, axis1=0, axis2=1) -> torch.tensor:
        n, d = features.shape
        if not (0 <= axis1 < d and 0 <= axis2 < d):
            raise ValueError("Axes must be valid dimensions within the feature vector.")
    
        # Compute cos and sin for each node
        cos_theta = torch.cos(angles).view(-1, 1, 1)  # Shape: (n, 1, 1)
        sin_theta = torch.sin(angles).view(-1, 1, 1)  # Shape: (n, 1, 1)
        
        # Construct batched rotation matrices (n, 2, 2)
        rotation_matrix = torch.zeros((n, 2, 2), device=self.device,dtype=torch.float32)
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
    

    #Override the forward method
    def forward(self,input:torch.tensor,A:torch.sparse.Tensor,x:torch.Tensor,batch:torch.tensor=None,dropout:float=None) -> torch.tensor:
        
        torch.cuda.empty_cache()
        
        if('zinc'==self.dataset_name.lower()):
             h = input.long() 
             h = self.embedding_h(h)
             input=F.dropout(h,dropout,training=self.training) #in_feature_dropout

        H=self.Linear(input) #Apply the linear layer on the features
        
        support=F.relu(H)    #Apply the relu activation
        
        support1=self.layernorm(support)  #Apply the layer norm on the features
        
        input=F.dropout(support1,self.dropout,training=self.training) #Applying dropout on the features
        
        coord=x #Set the coordinate variable
        
        #updated_h=torch.zeros_like(input,dtype=torch.float32,device=self.device)

        for i in range(self.L):
           
           theta_f, theta_x = self.Compute_Angular_Distance(input,coord) #Fetch the angular distance for both features and coordinates
           
           m=self.Compute_Message_Vector(A,input,theta_f,theta_x) #Get the messsage vector for individual edges
           
           coord=self.Coordinate_Update(coord,m,A) #Update the coordinates associated with the nodes
            
           aggregated_angle=self.Compute_Aggregated_Angle(coord,A) #Compute the aggregated angle for feature vector rotation for every node
           
           H_new=self.rotate_features_with_matrix(input, aggregated_angle, axis1=0, axis2=1)  #Rotate the feature vectors of the nodes using the aggregated angles
           #H_new=self.rotate_features_with_matrix2(input, aggregated_angle)

           #mask = (torch.rand_like(H_new) > 0.1).float() 
           
           #updated_h += (H_new * mask) * (1/self.L)

           input = H_new #Updated Feature Vector To Be Passed To The New Layer
           
        #H_new=updated_h  
           
        if('reg'==self.flag and 'mean'==self.readout.lower()):
            H_out=self.node_dec(H_new) #Pass the rotated features through the Node Level Feature Enhancement MLP
            
            H_out_pooled=global_mean_pool(H_out,batch) #Add pooling the features for graph level prediction #~(batch_size*d)
           
            pred = self.graph_dec(H_out_pooled)  #Pass the pooled features through decoder MLP
    
            return pred.squeeze(1)    #Return the final prediction as scaler

        
        elif('reg'==self.flag and 'add'==self.readout.lower()):

            H_out=self.node_dec(H_new) #Pass the rotated features through the Node Level Feature Enhancement MLP
            
            H_out_pooled=global_add_pool(H_out,batch) #Add pooling the features for graph level prediction #~(batch_size*d)
           
            pred = self.graph_dec(H_out_pooled)  #Pass the pooled features through decoder MLP
    
            return pred.squeeze(1)    #Return the final prediction as scaler
        
        else:
            H_out=self.classifier(H_new) #Pass the rotated features through the classifier MLP
            
            return H_out



