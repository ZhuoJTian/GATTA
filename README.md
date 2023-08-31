# GATTA
Implementation code for the GATTA algorithm

### Introduction
The code implements the GATTA algorithm in [Distributed Learning over Networks with Graph-Attention-Based Personalization](https://ieeexplore.ieee.org/document/10141684).

```
@ARTICLE{GATTA,
  author={Tian, Zhuojun and Zhang, Zhaoyang and Yang, Zhaohui and Jin, Richeng and Dai, Huaiyu},
  journal={IEEE Transactions on Signal Processing}, 
  title={Distributed Learning Over Networks With Graph-Attention-Based Personalization}, 
  year={2023},
  volume={71},
  pages={2071-2086},
  doi={10.1109/TSP.2023.3282071}}
```

### Requirements
1. `[Python 3.6 + TensorFlow 2.4.1 + CUDA 11.0] or [Python 2.7 + TensorFlow 1.15.0 + CUDA 10.0] ` (modify the ''import'' tensorflow.compat.v1 to tensorflow)
2. `numpy`
3. `tqdm`

### Usage

#### Part 1: CIFAR-10 on different numbers of labels

#### Part 2: FEMNIST on different numbers of writers

### Note
  - FEMNIST need to be preprocessed according to the official introduction and stored in `femnist/data/train` and `femnist/data/train`. Specifically, for the preprocessing, we shuffle the data and delete the users whose number of training samples are smaller than 10. Note there should be 3596 users left. Otherwise, modify the value in 82 line in the 'Dataset.py' from `FEMNIST_code/Model`. Then we separate the data for each user into 75% for training and 25% for testing and the results are stored in `FEMNIST/femnist/data/train` and `femnist/data/train`.

  - The data assignments for clients are already finished and stored. If one wants to reassign the non-i.i.d. data, run the 'Sample_parti_noiid3.py' in  `CIFAR10_code/Main`by uncommenting the code; or the 'Sample_parti_noiid2.py' in `FEMNIST_code/Main` .
