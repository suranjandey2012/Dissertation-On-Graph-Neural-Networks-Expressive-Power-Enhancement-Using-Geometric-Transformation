import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularGeometricConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_x = nn.Linear(in_features, out_features)
        self.linear_h = nn.Linear(in_features, out_features)
        
        # Geometric transformation layers
        self.geometric_transform = nn.Sequential(
            nn.Linear(out_features * 2 + 4, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        
    def compute_geometric_features(self, x, h):
        # Compute advanced geometric features
        dist_matrix = torch.cdist(x, x)  # Interatomic distances
        angle_matrix = self.compute_angle_matrix(x)
        
        return dist_matrix, angle_matrix
    
    def compute_angle_matrix(self, x):
        # Compute angle matrix between coordinates
        x_normed = F.normalize(x, dim=-1)
        cos_angles = torch.matmul(x_normed, x_normed.transpose(-1, -2))
        angle_matrix = torch.arccos(torch.clamp(cos_angles, -1+1e-6, 1-1e-6))
        return angle_matrix
    
    def forward(self, x, h):
        # Transform initial features
        x_transformed = self.linear_x(x)
        h_transformed = self.linear_h(h)
        
        # Compute geometric features
        dist_matrix, angle_matrix = self.compute_geometric_features(x, h)
        
        # Message passing with geometric awareness
        msg_features = torch.cat([
            h_transformed.unsqueeze(1).repeat(1, h.size(0), 1),  # Node features
            dist_matrix.unsqueeze(-1),  # Interatomic distances
            angle_matrix.unsqueeze(-1)  # Angular information
        ], dim=-1)
        
        # Geometric message transformation
        messages = self.geometric_transform(msg_features)
        
        return messages