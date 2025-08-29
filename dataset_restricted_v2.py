import os
from glob import glob
from datetime import datetime, timedelta

import numpy as np

import torch

from torch.utils.data import Dataset


class GK2ADataset(Dataset):
    def __init__(self, data_path='data',
                 reduce_size=1, T=6, T_interval=10,
                 transform=None):
        self.data_path = data_path
        self.reduce_size = reduce_size
        self.T = T
        self.T_interval = T_interval
        self.DATA_T_INTERVAL = 10
        self.transform = transform

        self._load_data_files()
        self._get_timestamps()
        self._make_dataset_file_mat()

    def _load_data_files(self):
        # self.files = sorted(glob(f"{self.data_path}/"
        #                          f"GK2A_reduceX{self.reduce_size}_new/**/*.pt",
        #                     recursive=True))
        self.files1 = glob(f"{self.data_path}/**/*ir105*.pt", recursive=True)
        self.files2 = glob(f"{self.data_path}/**/*sw038*.pt", recursive=True)
        self.files3 = glob(f"{self.data_path}/**/*wv069*.pt", recursive=True)
        self.files4 = glob(f"{self.data_path[:-2]+'v3'}/**/*.pt", recursive=True)
        self.files = np.array(sorted(self.files1 + self.files2 + self.files3 +  self.files4))
        
        self.files1 = np.array(sorted(self.files1))
        self.files2 = np.array(sorted(self.files2))
        self.files3 = np.array(sorted(self.files3))
        self.files4 = np.array(sorted(self.files4))
        # self.files = np.array(self.files)
    
    
    def _get_timestamps(self):
        timestamps = []
        for file in self.files:
            path, filename = os.path.split(file)
            ts = filename.split('_')[-1].split('.')[0]
            ts = datetime.strptime(ts, '%Y%m%d%H%M')
            timestamps.append(ts)

        # timestamp 기준 정렬
        order = np.argsort(timestamps)
        timestamps = np.array([timestamps[i] for i in order], dtype=object)
        files = self.files[order]

        # 채널 키워드(소문자)
        ch_tokens = {
            "ir105": "files1",
            "sw038": "files2",
            "wv069": "files3",
            "wv073": "files4",
        }

        # timestamp -> 파일 경로 (채널별) 매핑 만들기 (동일 ts에 같은 채널이 여러개면 처음 것만 사용)
        maps = {k: {} for k in ch_tokens}  # ex) maps["ir105"][ts] = file
        for f, ts in zip(files, timestamps):
            fl = f.lower()
            for token in ch_tokens:
                if token in fl and ts not in maps[token]:
                    maps[token][ts] = f

        # 네 채널 모두 존재하는 timestamp만 교집합으로 선택
        ts_sets = [set(maps[token].keys()) for token in ch_tokens]
        common_ts = sorted(set.intersection(*ts_sets))  # 모두 있는 시각만

        # 최종 결과 구성: 동일한 timestamp 순서로 각 채널 파일 배열 만들기
        self.timestamps = np.array(common_ts, dtype=object)
        self.files1 = np.array([maps["ir105"][ts] for ts in common_ts])
        self.files2 = np.array([maps["sw038"][ts] for ts in common_ts])
        self.files3 = np.array([maps["wv069"][ts] for ts in common_ts])
        self.files4 = np.array([maps["wv073"][ts] for ts in common_ts])
        
        
    def _make_dataset_file_mat(self):
        idx_interval = int(self.T_interval // self.DATA_T_INTERVAL)

        full_timestamp_data_indices = []
        try:
            for data_idx, timestamp in enumerate(self.timestamps):
                target_indices = data_idx + idx_interval * np.arange(2*self.T)
                data_timestamps = self.timestamps[target_indices]

                timedeltas = np.array([timedelta(minutes=t*self.T_interval) for t in range(2*self.T)])
                full_timestamps = timestamp + timedeltas

                if np.all(data_timestamps == full_timestamps):
                    full_timestamp_data_indices.append(target_indices)
        except:
            pass

        dataset_idx_mat = np.array(full_timestamp_data_indices)
        self.dataset_idx_mat = dataset_idx_mat
        self.data_file_mat1 = self.files1[dataset_idx_mat.astype(int)]
        self.data_file_mat2 = self.files2[dataset_idx_mat.astype(int)]
        self.data_file_mat3 = self.files3[dataset_idx_mat.astype(int)]
        self.data_file_mat4 = self.files4[dataset_idx_mat.astype(int)]
        
    def __getitem__(self, idx):
        video_files1 = self.data_file_mat1[idx]
        video_files2 = self.data_file_mat2[idx]
        video_files3 = self.data_file_mat3[idx]
        video_files4 = self.data_file_mat4[idx]

        video = []
        for file1, file2, file3, file4 in zip(video_files1,video_files2,video_files3,video_files4):
            
            frame = [torch.load(file1),torch.load(file2),torch.load(file3),torch.load(file4)]
            
            video.append(frame)
        video = torch.from_numpy(np.array(video)).squeeze(2)        
        
        x, y = video[:self.T], video[self.T:]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

    def __len__(self):
        return len(self.data_file_mat1)

class GK2ADataset_MultiSteps(Dataset):
    def __init__(self, data_path='data',
                 reduce_size=1, T=6, T_interval=10,
                 transform=None,
                 steps=1):  # >>> MODIFIED: add steps (# of T-blocks to predict)
        self.data_path = data_path
        self.reduce_size = reduce_size
        self.T = T
        self.steps = steps            # >>> MODIFIED
        self.T_interval = T_interval
        self.DATA_T_INTERVAL = 10
        self.transform = transform

        self._load_data_files()
        self._get_timestamps()
        self._make_dataset_file_mat()  # now builds (1+steps)*T sequences

    def _load_data_files(self):
        self.files1 = glob(f"{self.data_path}/**/*ir105*.pt", recursive=True)
        self.files2 = glob(f"{self.data_path}/**/*sw038*.pt", recursive=True)
        self.files3 = glob(f"{self.data_path}/**/*wv069*.pt", recursive=True)
        self.files4 = glob(f"{self.data_path[:-2]+'v3'}/**/*.pt", recursive=True)
        self.files = np.array(sorted(self.files1 + self.files2 + self.files3 + self.files4))

        self.files1 = np.array(sorted(self.files1))
        self.files2 = np.array(sorted(self.files2))
        self.files3 = np.array(sorted(self.files3))
        self.files4 = np.array(sorted(self.files4))

    def _get_timestamps(self):
        timestamps = []
        for file in self.files:
            path, filename = os.path.split(file)
            ts = filename.split('_')[-1].split('.')[0]
            ts = datetime.strptime(ts, '%Y%m%d%H%M')
            timestamps.append(ts)

        order = np.argsort(timestamps)
        timestamps = np.array([timestamps[i] for i in order], dtype=object)
        files = self.files[order]

        ch_tokens = {
            "ir105": "files1",
            "sw038": "files2",
            "wv069": "files3",
            "wv073": "files4",
        }

        maps = {k: {} for k in ch_tokens}
        for f, ts in zip(files, timestamps):
            fl = f.lower()
            for token in ch_tokens:
                if token in fl and ts not in maps[token]:
                    maps[token][ts] = f

        ts_sets = [set(maps[token].keys()) for token in ch_tokens]
        common_ts = sorted(set.intersection(*ts_sets))

        self.timestamps = np.array(common_ts, dtype=object)
        self.files1 = np.array([maps["ir105"][ts] for ts in common_ts])
        self.files2 = np.array([maps["sw038"][ts] for ts in common_ts])
        self.files3 = np.array([maps["wv069"][ts] for ts in common_ts])
        self.files4 = np.array([maps["wv073"][ts] for ts in common_ts])

    def _make_dataset_file_mat(self):
        # >>> MODIFIED: validate cadence & build length = (1+steps)*T
        if self.T_interval <= 0:
            raise ValueError(f"T_interval must be positive; got {self.T_interval}")
        if self.T_interval % self.DATA_T_INTERVAL != 0:
            raise ValueError(
                f"T_interval ({self.T_interval}) must be a multiple of DATA_T_INTERVAL "
                f"({self.DATA_T_INTERVAL})."
            )

        idx_interval = self.T_interval // self.DATA_T_INTERVAL
        step_count = (1 + self.steps) * self.T   # >>> MODIFIED: previously 2*T

        max_start = len(self.timestamps) - idx_interval * (step_count - 1)
        # if max_start <= 0:
        #     raise ValueError(
        #         "No room to build sequences: increase data or reduce T/steps.\n"
        #         f"N={len(self.timestamps)}, T={self.T}, steps={self.steps}, "
        #         f"T_interval={self.T_interval}, idx_step={idx_interval}"
        #     )

        delta = timedelta(minutes=self.T_interval)
        good_indices = []
        for start in range(max_start):
            idxs = start + idx_interval * np.arange(step_count)
            ts_seq = self.timestamps[idxs]
            # verify exact cadence between consecutive timestamps
            if all((ts_seq[i+1] - ts_seq[i]) == delta for i in range(step_count - 1)):
                good_indices.append(idxs)

        if not good_indices:
            raise ValueError(
                "No valid sequences found at the requested cadence.\n"
                f"T={self.T}, steps={self.steps}, T_interval={self.T_interval}"
            )

        dataset_idx_mat = np.stack(good_indices, axis=0)  # [N_seq, step_count]
        self.dataset_idx_mat = dataset_idx_mat
        self.data_file_mat1 = self.files1[dataset_idx_mat.astype(int)]
        self.data_file_mat2 = self.files2[dataset_idx_mat.astype(int)]
        self.data_file_mat3 = self.files3[dataset_idx_mat.astype(int)]
        self.data_file_mat4 = self.files4[dataset_idx_mat.astype(int)]

    def __getitem__(self, idx):
        video_files1 = self.data_file_mat1[idx]  # length = (1+steps)*T
        video_files2 = self.data_file_mat2[idx]
        video_files3 = self.data_file_mat3[idx]
        video_files4 = self.data_file_mat4[idx]

        # >>> MODIFIED: build tensor purely in torch (no numpy object array)
        frames = []
        for f1, f2, f3, f4 in zip(video_files1, video_files2, video_files3, video_files4):
            t1 = torch.load(f1)
            t2 = torch.load(f2)
            t3 = torch.load(f3)
            t4 = torch.load(f4)
            # ensure shape [C,H,W] for each time step; stack channels along C
            # if loaded tensors are [H,W], unsqueeze to [1,H,W]
            if t1.dim() == 2: t1 = t1.unsqueeze(0)
            if t2.dim() == 2: t2 = t2.unsqueeze(0)
            if t3.dim() == 2: t3 = t3.unsqueeze(0)
            if t4.dim() == 2: t4 = t4.unsqueeze(0)
            frame = torch.cat([t1, t2, t3, t4], dim=0)  # [4, H, W]
            frames.append(frame)
        video = torch.stack(frames, dim=0)  # [step_count, 4, H, W]

        # >>> MODIFIED: split into x (first T) and y (next steps*T)
        x = video[:self.T]                                  # [T, 4, H, W]
        y = video[self.T : self.T + self.steps * self.T]    # [steps*T, 4, H, W]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y

    def __len__(self):
        return len(self.dataset_idx_mat)  # >>> MODIFIED
