"""
stgcn.py

A PyTorch implementation of the Spatio-Temporal Graph Convolutional Network (STGCN)
with an adaptive graph convolutional layer as described in:

  A Spatio-Temporal Graph Convolutional Network Model for Internet of Medical Things (IoMT)
  Ghosh, D.K.; Chakrabarty, A.; Moon, H.; Piran, M.J.
  Sensors 2022, 22, 8438. https://doi.org/10.3390/s22218438

and its adaptive graph convolution design inspired by:

  Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition
  Shi, L.; Zhang, Y.; Cheng, J.; Lu, H.
  CVPR 2019.

The model assumes input skeleton data of shape (B, 3, 300, 25) (3 coordinates, 300 frames, 25 joints).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Adaptive Graph Convolutional Layer
# ---------------------------
def build_ntu_rgbd_adjacency(num_joints=25):
    """
    Build the fixed physical adjacency matrix for the NTU RGB+D dataset.
        
    Edges (bi-directional):
      0: Spine Base
      1: Spine
      20: Spine Shoulder
      2: Neck
      3: Head
      4: Shoulder Left
      5: Elbow Left
      6: Wrist Left
      7: Hand Left
      8: Shoulder Right
      9: Elbow Right
      10: Wrist Right
      11: Hand Right
      12: Hip Left
      13: Knee Left
      14: Ankle Left
      15: Foot Left
      16: Hip Right
      17: Knee Right
      18: Ankle Right
      19: Foot Right
      21: Hand Tip Left
      22: Thumb Left
      23: Hand Tip Right
      24: Thumb Right
    """
    # Define the list of edges (zero-indexed)
    edges = [
        (0, 1),    # Spine Base - Spine
        (1, 20),   # Spine - Spine Shoulder
        (20, 2),   # Spine Shoulder - Neck
        (2, 3),    # Neck - Head

        (20, 4),   # Spine Shoulder - Shoulder Left
        (4, 5),    # Shoulder Left - Elbow Left
        (5, 6),    # Elbow Left - Wrist Left
        (6, 7),    # Wrist Left - Hand Left

        (20, 8),   # Spine Shoulder - Shoulder Right
        (8, 9),    # Shoulder Right - Elbow Right
        (9, 10),   # Elbow Right - Wrist Right
        (10, 11),  # Wrist Right - Hand Right

        (0, 12),   # Spine Base - Hip Left
        (12, 13),  # Hip Left - Knee Left
        (13, 14),  # Knee Left - Ankle Left
        (14, 15),  # Ankle Left - Foot Left

        (0, 16),   # Spine Base - Hip Right
        (16, 17),  # Hip Right - Knee Right
        (17, 18),  # Knee Right - Ankle Right
        (18, 19),  # Ankle Right - Foot Right

        (7, 21),   # Hand Left - Hand Tip Left
        (7, 22),   # Hand Left - Thumb Left
        (11, 23),  # Hand Right - Hand Tip Right
        (11, 24)   # Hand Right - Thumb Right
    ]
    
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    # Add self-connections.
    for i in range(num_joints):
        A[i, i] = 1
    return torch.tensor(A)

# ---------------------------
# Adaptive Graph Convolutional Layer
# ---------------------------
class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints, 
                 kernel_size=3, embed_channels=None, A=None, residual=True):
        """
        Adaptive Graph Convolutional Layer exactly as described in the paper.
        
        Implements:
            f_out = sum_{k=1}^{K_v} W_k [ f_in * (A_k + B_k + C_k) ]
        where:
          - K_v (kernel_size) is the number of partitions (typically 3).
          - A_k: fixed physical adjacency for partition k.
          - B_k: learnable adjacency tensor of shape (K_v, num_joints, num_joints),
                 initialized to zero.
          - C_k: data-dependent adjacency computed using a normalized embedded Gaussian
                 function via two embeddings θ and φ.
          - W_k: realized as a 1×1 convolution applied after aggregation.
          - A residual connection is applied (with a 1×1 conv if needed).
          
        Args:
          in_channels: Number of input channels.
          out_channels: Number of output channels.
          num_joints: Number of joints (vertices), e.g., 25.
          kernel_size: Number of partitions K_v; typically 3.
          embed_channels: Embedding dimension (if None, defaults to max(in_channels // 2, 1)).
          A: Fixed physical adjacency tensor of shape (num_joints, num_joints).  
             If None, defaults to an identity matrix repeated kernel_size times.
          residual: Whether to use a residual connection.
        """
        super(AdaptiveGraphConv, self).__init__()
        self.num_joints = num_joints
        self.K_v = kernel_size
        self.residual = residual
        if embed_channels is None:
            embed_channels = max(in_channels // 2, 1)
        self.embed_channels = embed_channels
        
        # Define embedding functions θ and φ (1×1 convolutions).
        self.theta = nn.Conv2d(in_channels, embed_channels, kernel_size=1)
        self.phi   = nn.Conv2d(in_channels, embed_channels, kernel_size=1)
        
        # Learnable parameter B_k: shape (K_v, num_joints, num_joints), initialized to zero.
        self.B = nn.Parameter(torch.zeros(self.K_v, num_joints, num_joints), requires_grad=True)
        A = build_ntu_rgbd_adjacency(num_joints=25)

        # Fixed physical adjacency A_k: shape (K_v, num_joints, num_joints)
        if A is None:
            # If not provided, default to identity for each partition.
            print("A is none")
            A = torch.eye(num_joints).unsqueeze(0).repeat(self.K_v, 1, 1)
        else:
            # print("else")
            if A.dim() == 2:
                A = A.unsqueeze(0).repeat(self.K_v, 1, 1)
        self.register_buffer('A', A)
        # 1×1 convolution to represent the weighting function W.
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Residual connection: if the input and output channels differ, apply a 1x1 conv.
        if self.residual:
            if in_channels != out_channels:
                self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.residual_conv = None
        else:
            self.residual_conv = None
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x has shape: (B, C_in, T, N)
        BATCH, C, T, N = x.size()
        
        # Compute the embeddings.
        # f_theta and f_phi: (B, embed_channels, T, N)
        f_theta = self.theta(x)
        f_phi   = self.phi(x)
        
        # Permute so that the joint dimension (N) comes next: (B, N, embed_channels, T)
        f_theta = f_theta.permute(0, 3, 1, 2).contiguous()
        f_phi   = f_phi.permute(0, 3, 1, 2).contiguous()
        # Flatten the last two dimensions: shape (B, N, embed_channels * T)
        f_theta_flat = f_theta.view(BATCH, N, -1)
        f_phi_flat   = f_phi.view(BATCH, N, -1)
        
        # Compute the data-dependent component using the dot product and softmax.
        C_adapt = torch.bmm(f_theta_flat, f_phi_flat.transpose(1, 2))  # (B, N, N)
        C_adapt = F.softmax(C_adapt, dim=-1)
        # Replicate C_adapt over K_v partitions: shape (B, K_v, N, N)
        C_k = C_adapt.unsqueeze(1).expand(BATCH, self.K_v, N, N)
        
        # Combine the fixed physical graph A_k and learnable B_k.
        # Both have shape (K_v, N, N).
        A_combined = self.A + self.B  # (K_v, N, N)
        A_combined = A_combined.unsqueeze(0)  # (1, K_v, N, N)
        
        # Form the final adaptive adjacency by adding the data-dependent component.
        # Adaptive_adj: (B, K_v, N, N)
        Adaptive_adj = A_combined + C_k
        
        # Expand x to have a partition dimension:
        # x: (B, C, T, N) -> (B, 1, C, T, N) -> (B, K_v, C, T, N)
        x_expanded = x.unsqueeze(1).expand(BATCH, self.K_v, C, T, N)
        
        # Aggregate the features via the adaptive adjacency matrix using Einstein summation.
        # x_aggregated[b, k, c, t, m] = sum_{n} x[b, k, c, t, n] * Adaptive_adj[b, k, n, m]
        x_aggregated = torch.einsum('bkcTN,bkNM->bkcTM', x_expanded, Adaptive_adj)
        # Sum over the partition dimension.
        x_sum = torch.sum(x_aggregated, dim=1)  # shape: (B, C, T, N)
        
        # Apply the 1×1 convolution (weighting function).
        out = self.conv1x1(x_sum)
        
        # Add the residual connection.
        if self.residual:
            res = x if self.residual_conv is None else self.residual_conv(x)
            out = out + res
        
        out = self.bn(out)
        out = self.relu(out)
        return out

# ---------------------------
# Temporal Convolution
# ---------------------------
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(kernel_size // 2, 0))
    def forward(self, x):
        return self.conv(x)

# ---------------------------
# STGCN Block with Adaptive Spatial GCN
# ---------------------------
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints=25,
                 spatial_kernel_size=3, temporal_kernel_size=9, residual=True):
        super(STGCNBlock, self).__init__()
        # Adaptive Graph Convolution (Spatial branch)
        self.adaptive_spatial = AdaptiveGraphConv(in_channels, out_channels,
                                                    num_joints=num_joints,
                                                    kernel_size=spatial_kernel_size,
                                                    residual=residual)
        # Temporal Convolution branch
        self.temporal_conv = TemporalConv(in_channels, out_channels,
                                          kernel_size=temporal_kernel_size)
        # Fuse branches: concatenate along channel dimension then reduce with 1x1 conv.
        self.fuse_conv = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)
        # Residual connection across the block (if input dims differ, adjust)
        if residual:
            if in_channels != out_channels:
                self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.residual_conv = None
        else:
            self.residual_conv = None
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: (B, C, T, N)
        spatial_out = self.adaptive_spatial(x)   # (B, out_channels, T, N)
        temporal_out = self.temporal_conv(x)       # (B, out_channels, T, N)
        # Concatenate along channels
        cat = torch.cat([spatial_out, temporal_out], dim=1)  # (B, 2*out_channels, T, N)
        fused = self.fuse_conv(cat)  # (B, out_channels, T, N)
        # Residual connection
        if self.residual_conv is not None:
            res = self.residual_conv(x)
        else:
            res = x
        out = fused + res
        out = self.bn(out)
        out = self.dropout(out)
        out = self.relu(out)

        return out

# ---------------------------
# Full STGCN Model
# ---------------------------
class STGCN(nn.Module):
    def __init__(self, num_classes, num_joints=25, num_frames=300):
        super(STGCN, self).__init__()
        # Input BatchNorm on raw data (3 channels)
        self.input_bn = nn.BatchNorm2d(3)
        # Define blocks (adapt the channel progression from the paper)
        # According to the paper: first 4 blocks: 64 channels, next 3 blocks: 128 channels, last 3: 256 channels.
        self.blocks = nn.ModuleList()
        layers = [(3, 64), (64, 64), (64, 64), (64, 64),
                  (64, 128), (128, 128), (128, 128),
                  (128, 256), (256, 256), (256, 256)]
        for in_c, out_c in layers:
            self.blocks.append(
                STGCNBlock(in_channels=in_c, out_channels=out_c,
                           num_joints=num_joints,
                           spatial_kernel_size=3,
                           temporal_kernel_size=9,
                           residual=True)
            )
        # Global average pooling over temporal and joint dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Final fully-connected layer: input channels = output of last block.
        self.fc = nn.Linear(layers[-1][1], num_classes)

    def forward(self, x):
        # x shape: (B, 3, T, N) - expected T = num_frames (e.g., 300) and N = num_joints (25)
        x = self.input_bn(x)
        for block in self.blocks:
            x = block(x)
        # Global average pooling over T and N
        x = self.global_avg_pool(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)     # (B, C)
        out = self.fc(x)              # (B, num_classes)
        return out

 # test
if __name__ == "__main__":
    model = STGCN(num_classes=60, num_joints=25, num_frames=300)
    dummy_input = torch.randn(8, 3, 300, 25)
    output = model(dummy_input)
    print("Output shape:", output.shape)
