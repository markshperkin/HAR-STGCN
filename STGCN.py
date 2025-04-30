from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# skeleton graph representation
# ---------------------------
def build_ntu_rgbd_matrix():
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
    return [
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
    
# ---------------------------
# Ak representation Ak = Λk−^(1/2) Ak¯ Λk−^(1/2)
# ---------------------------

def build_partitioned_adjacency(num_joints, edge_list, root=0, alpha=0.001):
    # build raw symmetric adjacency matrix
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edge_list:
        A[i, j] = 1
        A[j, i] = 1

    # compute shortest path distance from each node to the root
    dist = [np.inf] * num_joints
    dist[root] = 0
    q = deque([root])
    while q:
        v = q.popleft()
        for u in range(num_joints):
            if A[v, u] and dist[u] == np.inf:
                dist[u] = dist[v] + 1
                q.append(u)
    dist = np.array(dist, dtype=np.int32)

    # init the three raw partitions Abar_k
    Abar = np.zeros((3, num_joints, num_joints), dtype=np.float32)
    # self connections k=0
    for i in range(num_joints):
        Abar[0, i, i] = 1
    # inward edges (j one hop closer to root than i) -> k=1
    # outward edges (j one hop further from root than i) -> k=2
    for i in range(num_joints):
        for j in range(num_joints):
            if A[i, j] == 1:
                if dist[j] == dist[i] - 1:
                    Abar[1, i, j] = 1
                elif dist[j] == dist[i] + 1:
                    Abar[2, i, j] = 1

    # symmetric normalization
    A = np.zeros_like(Abar)
    for k in range(3):
        Ak = Abar[k]
        D = np.sum(Ak, axis=1) + alpha
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
        # normalized: D^{-1/2} A_k D^{-1/2}
        A[k] = D_inv_sqrt @ Ak @ D_inv_sqrt

    return torch.from_numpy(A)  # shape (3, N, N)


# ---------------------------
# adaptive graph convolutional layer
# ---------------------------
class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints, 
                 kernel_size=3, embed_channels=None, A=None, residual=True):
        super(AdaptiveGraphConv, self).__init__()
        self.num_joints = num_joints
        self.K_v = kernel_size
        self.residual = residual
        
        # define embedding functions (1×1 conv).
        self.theta = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.phi   = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # learnable parameter B_k, initialized to zero.
        self.B = nn.Parameter(torch.zeros(self.K_v, num_joints, num_joints), requires_grad=True)

        edges = build_ntu_rgbd_matrix()
        # normalized partitions
        A_partitioned = build_partitioned_adjacency(
            num_joints=num_joints,
            edge_list=edges,
            root=0,
            alpha= 0.001
        )
        self.register_buffer('A', A_partitioned)
        # W_k per partition
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(self.K_v)
        ])
        
        # residual connection. if the input and output channels differ, apply a 1x1 conv.
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
        
        f_theta = self.theta(x)
        f_phi   = self.phi(x)
        
        # Permute so that the joint dimension (N) comes next (B, N, embed channels, T)
        f_theta = f_theta.permute(0, 3, 1, 2).contiguous()
        f_phi   = f_phi.permute(0, 3, 1, 2).contiguous()
        # flatten the last two dimensions (B, N, embed_channels * T)
        f_theta_flat = f_theta.view(BATCH, N, -1)
        f_phi_flat   = f_phi.view(BATCH, N, -1)
        
        # compute the data dependent component using the dot product and softmax.
        C_adapt = torch.bmm(f_theta_flat, f_phi_flat.transpose(1, 2))
        C_adapt = F.softmax(C_adapt, dim=-1)
        # replicate C adapt over K_v partitions (B, K_v, N, N)
        C_k = C_adapt.unsqueeze(1).expand(BATCH, self.K_v, N, N)
        
        # combine the fixed physical graph A_k and learnable B_k.
        A_combined = self.A + self.B
        A_combined = A_combined.unsqueeze(0)
        
        # form final adaptive adjacency by adding the data dependent component
        # Adaptive_adj: (B, K_v, N, N)
        Adaptive_adj = A_combined + C_k
        
        # expand x to have a partition dimension:
        # (B, C, T, N) -> (B, 1, C, T, N) -> (B, K_v, C, T, N)
        x_expanded = x.unsqueeze(1).expand(BATCH, self.K_v, C, T, N)
        
        # aggregate the features via the adaptive adjacency matrix using Einstein summation
        # x_aggregated[b, k, c, t, m] = sum_{n} x[b, k, c, t, n] * adaptive_adj[b, k, n, m]
        x_aggregated = torch.einsum('bkcTN,bkNM->bkcTM', x_expanded, Adaptive_adj)
        # sum over the partition dimension.
        
        # apply 1×1 conv (weighting function).
        # new: apply each W_k before summing
        out = 0
        for k, conv in enumerate(self.convs):
            # x_agg_k is the aggregated features for partition k
            out = out + conv(x_aggregated[:, k])
        # now out has shape (B, out channels, T, N)
        
        # add the residual connection
        if self.residual:
            res = x if self.residual_conv is None else self.residual_conv(x)
            out = out + res
        
        out = self.bn(out)
        out = self.relu(out)
        return out

# ---------------------------
# temporal convolution
# ---------------------------
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(kernel_size // 2, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# ---------------------------
# STGCN block
# ---------------------------
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints=25,
                 spatial_kernel_size=3, temporal_kernel_size=9, residual=True):
        super(STGCNBlock, self).__init__()
        # adaptive graph conv
        self.adaptive_spatial = AdaptiveGraphConv(in_channels, out_channels,
                                                    num_joints=num_joints,
                                                    kernel_size=spatial_kernel_size,
                                                    residual=residual)
        # temporal conv
        self.temporal_conv = TemporalConv(in_channels, out_channels,
                                          kernel_size=temporal_kernel_size)
        # fuse branches. concat along channel dimension then reduce with 1x1 conv
        self.fuse_conv = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)
        # residual connection across the block (if input dims differ, adjust)
        if residual:
            if in_channels != out_channels:
                self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.residual_conv = None
        else:
            self.residual_conv = None
        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        spatial_out = self.adaptive_spatial(x)
        temporal_out = self.temporal_conv(x)
        # concat along channels
        cat = torch.cat([spatial_out, temporal_out], dim=1)
        fused = self.fuse_conv(cat)
        # residual connection
        if self.residual_conv is not None:
            res = self.residual_conv(x)
        else:
            res = x
        out = fused + res
        out = self.bn(out)
        out = self.relu(out)

        return out

# ---------------------------
# full STGCN model
# ---------------------------
class STGCN(nn.Module):
    def __init__(self, num_classes, num_joints=25, num_frames=300):
        super(STGCN, self).__init__()
        # batchNorm
        self.input_bn = nn.BatchNorm2d(3)
        # define blocks
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
        # global average pooling over temporal and joint dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # fc
        self.fc = nn.Linear(layers[-1][1], num_classes)

    def forward(self, x):
        x = self.input_bn(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

 # test with dummy input to see if working
if __name__ == "__main__":
    model = STGCN(num_classes=60, num_joints=25, num_frames=300)
    dummy_input = torch.randn(8, 3, 300, 25)
    output = model(dummy_input)
    print("Output shape:", output.shape)
