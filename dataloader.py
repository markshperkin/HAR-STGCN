import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit

class STGCNDataset(Dataset):
    """
    A custom dataset for STGCN training that:
      - Loads preprocessed .npy files from the "npydataset" folder.
      - Filters out samples belonging to classes A50-A60.
      - Extracts the "skel_body0" data.
      - If the number of frames T is less than 300, replicates the frames to pad to exactly 300.
      - Returns a tuple (data, label) where data has shape (3, 300, 25)
        and label is a zero-indexed integer.
        
    """
    def __init__(self, dataset_dir):
        super(STGCNDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.file_list = []
        self.labels = []
        self.label_pattern = re.compile(r'A(\d{3})')
        
        for filename in os.listdir(self.dataset_dir):
            if not filename.endswith('.npy'):
                continue
            match = self.label_pattern.search(filename)
            if not match:
                print(f"Warning: No action label found in filename: {filename}")
                continue
            action_label_str = match.group(1)
            action_label_int = int(action_label_str)
            # skip classes A50-A60 because they are two person data (to siplify implementation)
            if 50 <= action_label_int <= 60:
                continue
            full_path = os.path.join(self.dataset_dir, filename)
            try:
                data = np.load(full_path, allow_pickle=True).item()
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                continue
            if 'skel_body0' not in data:
                # skip files without key
                # print(f"Skipping file {full_path}: 'skel_body0' not found.")
                continue
            self.file_list.append(full_path)
            self.labels.append(action_label_int - 1)  # zero-indexed

    def __len__(self):
        return len(self.file_list)

    def _replicate_frames(self, tensor, target_frames=300):
        """
        Replicate frames in the tensor (shape: (C, T, V)) to reach target_frames.
        """
        C, T, V = tensor.shape
        if T >= target_frames:
            return tensor[:, :target_frames, :]
        reps = (target_frames + T - 1) // T  
        replicated = np.tile(tensor, (1, reps, 1))
        return replicated[:, :target_frames, :]

    def __getitem__(self, index):
        file_path = self.file_list[index]
        label = self.labels[index]
        try:
            data = np.load(file_path, allow_pickle=True).item()
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")
        
        skel = data['skel_body0']  # shape: (T, 25, 3)
        skel = np.array(skel, dtype=np.float32)
        # transpose from (T, 25, 3) to (3, T, 25)
        skel = np.transpose(skel, (2, 0, 1))
        data = self._replicate_frames(skel, target_frames=300)
        data = torch.from_numpy(data)
        label = torch.tensor(label, dtype=torch.long)
        return data, label

def get_dataloaders(dataset_dir, batch_size=16, train_split=0.8, val_split=0.2, seed=42):
    """
    Creates stratified train and validation DataLoaders (80/20 split) for the STGCN dataset.

    Parameters:
      - dataset_dir: directory where the .npy files are stored.
      - batch_size: batch size for the loaders.
      - train_split, val_split: fractions that sum to 1 (default 0.8 and 0.2).
      - seed: random seed for reproducible splits.

    Returns:
      - train_loader, val_loader
    """
    full_dataset = STGCNDataset(dataset_dir)
    labels = np.array(full_dataset.labels)
    indices = np.arange(len(full_dataset))

    # stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(sss.split(indices, labels))

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def print_class_distribution(dataset, dataset_name="Dataset"):

    distribution = defaultdict(int)
    full_dataset = dataset.dataset
    for idx in dataset.indices:
        label = full_dataset.labels[idx]
        distribution[label] += 1

    print(f"Class distribution for {dataset_name}:")
    for label, count in sorted(distribution.items()):
        print(f"  Class {label:02d}: {count} samples")
    print()

if __name__ == "__main__":
    dataset_root = os.path.join(os.getcwd(), "npydataset")
    
    train_loader, val_loader = get_dataloaders(dataset_root, batch_size=16)
    
    train_indices = train_loader.dataset.indices
    train_labels = [train_loader.dataset.dataset.labels[i] for i in train_indices]
    train_count_labels = len(train_labels)

    val_indices = val_loader.dataset.indices
    val_labels = [val_loader.dataset.dataset.labels[i] for i in val_indices]
    val_count_labels = len(val_labels)

    total_count_labels = train_count_labels + val_count_labels
    print("===== DataLoader Test =====")
    print(f"Total Samples: {len(train_loader.dataset) + len(val_loader.dataset)}")
    print(f"Total Labels (from split indices): {total_count_labels}")

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Training Labels: {train_count_labels}")

    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Validation Labels: {val_count_labels}")
    
    print_class_distribution(train_loader.dataset, "Training Set")
    print_class_distribution(val_loader.dataset, "Validation Set")
    
    for data, labels in train_loader:
        print("data shape:", data.shape)
        print("Labels shape:", labels.shape)
        break
