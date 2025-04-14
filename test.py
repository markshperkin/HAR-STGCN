import os
import torch
import torch.nn as nn
from STGCN import STGCN 
from dataloader import get_dataloaders

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    print(f"Using device: {device}")

def test_model(model, test_loader):
    """
    Evaluate the model on the test dataset.
    Returns the overall accuracy.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for spatial_data, labels in test_loader:
            inputs = spatial_data.to(device)  
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples

def main():
    dataset_dir = os.path.join(os.getcwd(), "npydataset")
    batch_size = 16

    _, _, test_loader = get_dataloaders(dataset_dir, batch_size=batch_size)
    
    model = STGCN(num_classes=50, num_joints=25, num_frames=300).to(device)
    
    model_path = "best_stgcn_model2.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    test_acc = test_model(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
