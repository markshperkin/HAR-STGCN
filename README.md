
# Spatial-Temporal Graph Convolution Network
## Overview
This is my final project for [Edge and Neuromorphic Computing class](https://cse.sc.edu/class/714) where I implemented [A Spatio-Temporal Graph Convolutional Network Model for Internet of Medical Things (IoMT)](https://www.mdpi.com/1424-8220/22/21/84386) architecture for classifying human activity recognition 3D skeletal dataset into 49 different categories from The [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) dataset. 
for more information, please see my final report.
---
## how to run:
### Clone The Repo:
```bash
git clone https://github.com/markshperkin/HAR-STGCN.git
cd HAR-STGCN
```

### *RECOMMENDED* install and use Cuda for Nvidia GPU

### Download the dataset by following the [link](https://rose1.ntu.edu.sg/dataset/actionRecognition/) instructions. in this project, I used NTU RGB+D with 60 classes, only the 3D skeletal data. 

### Once dataset is downloaded, clone this repo and run a script who will convert the dataset into npy files
```bash
git clone https://github.com/shahroudy/NTURGB-D/tree/master/Python
cd NTURGB-D/Python/
python .\txt2npy.py
```
## Once dataset is downloaded and converted to .npy:
### Count the samples in the dataset
```bash
python .\count_samples.py
```

### Count samples produced by the dataloader
```bash
python .\dataloader.py
```

### Visualize a sample
```bash
python .\testdata.py
```

### Test if model is working, see output shape
```bash
python .\STGCN.py
```

### Start training the model. this is begin training for 50 epochs, could take around 16 hours for 49 classes. will output a csv file with all the training data
```bash
python .\train.py
```

### Plot the training data
```bash
python .\plot_results.py
```

### Test the latency of the trained model
```bash
python .\testLatency.py
```
---
## Class Project

This project was developed as part of the [Edge and Neuromorphic Computing class](https://cse.sc.edu/class/714) under the instruction of [Professor Ramtin Zand](https://sc.edu/study/colleges_schools/engineering_and_computing/faculty-staff/zand.php) at the University of South Carolina.


