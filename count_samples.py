import os
import re
import numpy as np
from collections import defaultdict

dataset_dir = os.path.join(os.getcwd(), "npydataset")

action_counts      = defaultdict(int)
action_time_totals = defaultdict(int)
action_max_frames  = defaultdict(lambda: 0)

# regular expression to capture the action label in the filename.
action_pattern = re.compile(r'A(\d{3})')

for filename in os.listdir(dataset_dir):
    if not filename.endswith('.npy'):
        continue

    match = action_pattern.search(filename)
    if not match:
        print(f"no action label found in filename: {filename}")
        continue
    action_label = match.group(1)
    action_counts[action_label] += 1

    file_path = os.path.join(dataset_dir, filename)
    try:
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    if 'skel_body0' in data:
        nframe = data['skel_body0'].shape[0]
    else:
        nframe = 0

    action_time_totals[action_label] += nframe

    if nframe > action_max_frames[action_label]:
        action_max_frames[action_label] = nframe

print("Action Class Summary:")
header = f"{'Action':<8} {'Samples':>8} {'Total Frames':>15} {'Avg Frames/Sample':>20} {'Max Frames':>15}"
print(header)
for i in range(1, 61):
    label = f"{i:03d}"
    count = action_counts.get(label, 0)
    total_frames = action_time_totals.get(label, 0)
    avg_frames = total_frames / count if count else 0
    max_frames = action_max_frames.get(label, 0)
    print(f"A{label:<6} {count:8d} {total_frames:15d} {avg_frames:20.2f} {max_frames:15d}")
