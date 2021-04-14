import os
import os.path as osp
import math
import random
import pickle
import warnings

import glob
import h5py

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.datasets.video_utils import VideoClips
import pytorch_lightning as pl


class VideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, sequence_length, train=True, resolution=64):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        folder = osp.join(data_folder, 'train' if train else 'test')
        files = sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])

        # hacky way to compute # of classes (count # of unique parent directories)
        self.classes = list(set([get_parent_dir(f) for f in files]))
        self.classes.sort()
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        warnings.filterwarnings('ignore')
        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
        if not osp.exists(cache_file):
            clips = VideoClips(files, sequence_length, num_workers=32)
            pickle.dump(clips.metadata, open(cache_file, 'wb'))
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(files, sequence_length,
                               _precomputed_metadata=metadata)
        self._clips = clips

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        resolution = self.resolution
        video, _, _, idx = self._clips.get_clip(idx)

        class_name = get_parent_dir(self._clips.video_paths[idx])
        label = self.class_to_label[class_name]
        return dict(video=preprocess(video, resolution), label=label)


def get_parent_dir(path):
    return osp.basename(osp.dirname(path))


def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW

    video -= 0.5

    return video


class HDF5Dataset(data.Dataset):
    """ Generic dataset for data stored in h5py as uint8 numpy arrays.
    Reads videos in {0, ..., 255} and returns in range [-0.5, 0.5] """
    def __init__(self, data_file, sequence_length, train=True, resolution=64):
        """
        Args:
            data_file: path to the pickled data file with the
                following format:
                {
                    'train_data': [B, H, W, 3] np.uint8,
                    'train_idx': [B], np.int64 (start indexes for each video)
                    'test_data': [B', H, W, 3] np.uint8,
                    'test_idx': [B'], np.int64
                }
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        # read in data
        self.data = h5py.File(data_file, 'r')
        prefix = 'train' if train else 'test'
        self._images = self.data[f'{prefix}_data']
        self._idx = self.data[f'{prefix}_idx']

        # compute splits for all possible sequences
        self._splits = self._compute_seq_splits()

    @property
    def n_classes(self):
        raise Exception('class conditioning not support for HDF5Dataset')

    def _compute_seq_splits(self):
        splits = []
        n_videos = len(self._idx)
        for i in range(n_videos):
            start = self._idx[i]
            end = self._idx[i + 1] if i < n_videos - 1 else n_videos
            splits.extend([(start + i, start + i + self.sequence_length)
                           for i in range(end - start - self.sequence_length)])
        return splits

    def __len__(self):
        return len(self._splits)

    def __getitem__(self, idx):
        start_idx, end_idx = self._splits[idx]
        video = torch.tensor(self._images[start_idx:end_idx])
        return dict(video=preprocess(video, self.resolution))


class VideoData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.hparams = args

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes


    def _dataset(self, train):
        Dataset = VideoDataset if osp.isdir(self.hparams.data_path) else HDF5Dataset
        dataset = Dataset(self.hparams.data_path, self.hparams.sequence_length,
                          train=train, resolution=self.hparams.resolution)
        return dataset


    def _dataloader(self, train):
        dataset = self._dataset(train)
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()
