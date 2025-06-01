import torch
import torch.nn

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

