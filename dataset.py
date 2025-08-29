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
        self.files = sorted(glob(f"{self.data_path}/**/*.pt",
                    recursive=True))

        self.files = np.array(self.files)

    def _get_timestamps(self):
        timestamps = []
        for file in self.files:
            path, filename = os.path.split(file)
            timestamp = filename.split('_')[-1].split('.')[0]
            timestamp = datetime.strptime(timestamp, '%Y%m%d%H%M')
            timestamps.append(timestamp)
        self.timestamps = np.array(timestamps)

    def _make_dataset_file_mat(self):
        idx_interval = int(self.T_interval // self.DATA_T_INTERVAL)

        full_timestamp_data_indices = []
        try:
            for data_idx, timestamp in enumerate(self.timestamps):
                target_indices = data_idx + idx_interval * np.arange(2*self.T)
                data_timestamps = self.timestamps[target_indices]

                timedeltas = np.array([timedelta(minutes=t*self.T_interval)
                                       for t in range(2*self.T)])
                full_timestamps = timestamp + timedeltas

                if np.all(data_timestamps == full_timestamps):
                    full_timestamp_data_indices.append(target_indices)
        except: pass

        dataset_idx_mat = np.array(full_timestamp_data_indices)

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


if __name__ == '__main__':
    dataset = GK2ADataset(data_path='/home/work/boostcamp_data', T=4, T_interval=10)
    x, y = dataset[0]

    video = torch.concatenate([x, y], dim=0).squeeze()

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    im = ax.imshow(video[0], cmap="gray", animated=True)
    ax.axis("off")


    def update(frame_idx):
        im.set_array(video[frame_idx])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(video), interval=200, blit=True
    )

    plt.show()

    ani.save("animation.mp4", writer="ffmpeg", fps=10)























