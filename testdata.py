import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import STGCNDataset  

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# NTU-style skeleton edges (25 joints)
bones = [
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
# make edges
bones += [(j,i) for (i,j) in bones]

def main():
    dataset_dir = os.path.join(os.getcwd(), "npydataset")
    
    ds = STGCNDataset(dataset_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    data, label = next(iter(loader))
    skel = data.squeeze(0).numpy()
    x, y, z = skel
    
    # set up plot
    fig, ax = plt.subplots()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Sample label = {label.item()}")
    ax.axis("equal")
    lines = [ax.plot([], [], marker="o")[0] for _ in bones]
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    def update(frame):
        for line, (i1, i2) in zip(lines, bones):
            xs = [x[frame, i1], x[frame, i2]]
            ys = [y[frame, i1], y[frame, i2]]
            line.set_data(xs, ys)
        return lines
    
    ani = FuncAnimation(
        fig, update,
        frames=skel.shape[1],
        interval=50,
        blit=True
    )
    
    plt.show()

if __name__ == "__main__":
    main()
