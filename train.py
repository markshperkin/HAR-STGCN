import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import csv


from STGCN import STGCN
from dataloader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (data, labels) in enumerate(loader):

        inputs = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    elapsed = time.time() - start_time
    print(f"Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.4f}  Time: {elapsed:.1f}s")
    return epoch_loss, epoch_acc, elapsed

def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(loader):
            inputs = data.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    elapsed = time.time() - start_time
    print(f"Val Loss:   {epoch_loss:.4f}  Val Acc:   {epoch_acc:.4f}  Time: {elapsed:.1f}s")
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=[ 30, 40], gamma=0.1)

    best_val_acc = 0.0

    epoch_results = []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch [{epoch}/{num_epochs}]")
        train_loss, train_acc , epochTime = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_stgcn_model_49c.pth")
            print("-> Best model saved.")
        print("-" * 40)

        epoch_results.append([epoch, train_loss, train_acc, val_loss, val_acc, epochTime])


    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")

    with open("trainresults_remake.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Epoch Time (s)"])
        writer.writerows(epoch_results)

def main():
    dataset_dir = os.path.join(os.getcwd(), "npydataset")
    batch_size = 16

    train_loader, val_loader = get_dataloaders(dataset_dir, batch_size=batch_size)

    model = STGCN(num_classes=49, num_joints=25, num_frames=300).to(device)

    train_model(model, train_loader, val_loader, num_epochs=50)


if __name__ == "__main__":
    main()
