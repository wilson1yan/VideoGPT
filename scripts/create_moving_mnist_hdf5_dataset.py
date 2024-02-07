import numpy as np

from torchvision.datasets import MovingMNIST

import h5py

split_ratio=19

dataset = MovingMNIST(root='./datasets', split="train", download=True, split_ratio=split_ratio)

# Write the Moving MNIST dataset to a HDF5 file with the following format, without using .cumulative_sizes
#     {
#         'train_data': [B, H, W, 3] np.uint8,
#         'train_idx': [B], np.int64 (start indexes for each video)
#         'test_data': [B', H, W, 3] np.uint8,
#         'test_idx': [B'], np.int64
#     }
#     where B is the number of training videos, H and W are the height and width of the video frames, and 3 is the number of channels (RGB).
with h5py.File('datasets/moving_mnist.h5', 'w') as f:
    # Flatten along index and time dimensions for train_data
    train_data = dataset.data.view(-1, 64, 64, 1).numpy()
    # Duplicate single gray value axis to 3 channels

    train_data = np.repeat(train_data, 3, axis=-1)
    train_idx = np.arange(0, len(dataset) * split_ratio, split_ratio)
    f.create_dataset('train_data', data=train_data)
    f.create_dataset('train_idx', data=train_idx)

    # NOTE: for now, both train and test data are the same

    # Flatten along index and time dimensions for test_data
    test_data = dataset.data.view(-1, 64, 64, 1).numpy()
    # Duplicate single gray value axis to 3 channels
    test_data = np.repeat(test_data, 3, axis=-1)
    test_idx = np.arange(0, len(dataset) * split_ratio, split_ratio)
    f.create_dataset('test_data', data=test_data)
    f.create_dataset('test_idx', data=test_idx)