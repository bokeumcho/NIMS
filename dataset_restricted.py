import os
from glob import glob
from datetime import datetime, timedelta

import numpy as np

import torch

from torch.utils.data import Dataset


class GK2ADataset(Dataset):
    def __init__(self, data_path='data',
                 reduce_size=1, T=6, T_interval=10,
                 transform=None, video_type=['ir105','sw038','wv069']):
        self.data_path = data_path
        self.reduce_size = reduce_size
        self.T = T
        self.T_interval = T_interval
        self.DATA_T_INTERVAL = 10
        self.transform = transform
        self.video_type = video_type

        self._load_data_files()
        self._get_timestamps()
        self._make_dataset_file_mat()

    def _load_data_files(self):
        # self.files = sorted(glob(f"{self.data_path}/"
        #                          f"GK2A_reduceX{self.reduce_size}_new/**/*.pt",
        #                     recursive=True))
        self.files_dict = {}
        for type in self.video_type:
            files = sorted(glob(f"{self.data_path}/**/gk2a_ami_le1b_{self.video_type}*.pt",
                    recursive=True))
            self.files_dict[self.video_type] = np.array(files)

    def _get_timestamps(self):
        timestamps = []
        for type in self.video_type:
            for file in self.files_dict[type]:
                path, filename = os.path.split(file)
                ts = filename.split('_')[-1].split('.')[0]
                ts = datetime.strptime(ts, '%Y%m%d%H%M')
                timestamps.append(ts)
        # sort by timestamp (not just by path) and keep files aligned
        order = np.argsort(timestamps)
        self.timestamps = np.array([timestamps[i] for i in order], dtype=object)
        self.files = self.files[order]

    def _make_dataset_file_mat(self):
        if self.T_interval <= 0:
            raise ValueError(f"T_interval must be positive; got {self.T_interval}")
        if self.T_interval % self.DATA_T_INTERVAL != 0:
            raise ValueError(
                f"T_interval ({self.T_interval}) must be a multiple of raw cadence "
                f"DATA_T_INTERVAL ({self.DATA_T_INTERVAL}) since files are at fixed cadence."
            )

        idx_interval = self.T_interval // self.DATA_T_INTERVAL  # exact integer step
        step_count = 2 * self.T
        if step_count <= 1:
            raise ValueError(f"T must be >= 1; got {self.T}")

        # last start index that keeps the sequence within bounds
        max_start = len(self.timestamps) - idx_interval * (step_count - 1)
        if max_start <= 0:
            raise ValueError(
                "No room to build sequences: increase data, reduce T, "
                f"or reduce T_interval. N={len(self.timestamps)}, T={self.T}, "
                f"T_interval={self.T_interval}, step={idx_interval}"
            )

        indices = []
        delta = timedelta(minutes=self.T_interval)
        for start in range(max_start):
            idxs = start + idx_interval * np.arange(step_count)
            ts_seq = self.timestamps[idxs]

            # verify cadence: every successive pair must differ by exactly T_interval
            ok = True
            for i in range(step_count - 1):
                if (ts_seq[i+1] - ts_seq[i]) != delta:
                    ok = False
                    break
            if ok:
                indices.append(idxs)

        if not indices:
            raise ValueError(
                "No valid (2T)-length sequences found at the requested cadence. "
                "Likely missing frames or mismatched T_interval. "
                f"T={self.T}, T_interval={self.T_interval}."
            )

        dataset_idx_mat = np.stack(indices, axis=0)  # shape: (num_sequences, 2T)
        self.data_file_mat = self.files[dataset_idx_mat.astype(int)]

    def __getitem__(self, idx):
        video_files = self.data_file_mat[idx]

        video = []
        for file in video_files:
            frame = torch.load(file) #.unsqueeze(0)
            video.append(frame)
        video = torch.stack(video, dim=0) # (2T, 1, H, W)

        x, y = video[:self.T], video[self.T:]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

    def __len__(self):
        return len(self.data_file_mat)

