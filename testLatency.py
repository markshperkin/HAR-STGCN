import os
import time
import torch

from STGCN import STGCN
from dataloader import get_dataloaders

def measure_latency(model, loader, device, num_warmup_batches=5):
    model.to(device)
    model.eval()

    # warm up
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i >= num_warmup_batches:
                break
            inputs = data.to(device)
            _ = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    batch_times = []
    with torch.no_grad():
        for data, _ in loader:
            inputs = data.to(device)
            start = time.time()
            _ = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            batch_times.append(end - start)

    batch_times = torch.tensor(batch_times)
    avg_batch = batch_times.mean().item()
    std_batch = batch_times.std().item()
    bs = loader.batch_size
    avg_sample = avg_batch / bs
    throughput = bs / avg_batch

    print(f"\nDevice: {device}")
    print(f"Avg  atency / batch: {avg_batch:.6f} s  (±{std_batch:.6f})")
    print(f"Avg latency / sample: {avg_sample:.6f} s")
    print(f"Throughput: {throughput:.1f} samples/s")

def main():
    dataset_dir = os.path.join(os.getcwd(), "npydataset")
    batch_size  = 16
    model_path  = "best_stgcn_model_49c.pth"
    num_classes = 49
    num_joints  = 25
    num_frames  = 300

    _, val_loader = get_dataloaders(dataset_dir, batch_size=batch_size)

    state_dict = torch.load(model_path, map_location='cpu')

    model_cpu = STGCN(num_classes=num_classes,
                      num_joints=num_joints,
                      num_frames=num_frames)
    model_cpu.load_state_dict(state_dict)

    # measure on CPU
    cpu_device = torch.device("cpu")
    measure_latency(model_cpu, val_loader, cpu_device)

    # measure on GPU
    model_gpu = STGCN(num_classes=num_classes,
                        num_joints=num_joints,
                        num_frames=num_frames)
    model_gpu.load_state_dict(state_dict)
    gpu_device = torch.device("cuda")
    measure_latency(model_gpu, val_loader, gpu_device)

if __name__ == "__main__":
    main()
