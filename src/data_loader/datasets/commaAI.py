
import os
import torch
import torchvision
import numpy as np
from tools.lib.framereader import FrameReader # Checkout https://docs.comma.ai/modules.html# on how to setup comma.ai tools


class CommaAI(torch.utils.data.Dataset):

    def __init__(self, config, mode, frame_transform, video_transform=None):
        super(CommaAI).__init__()

        self.mode = mode
        self.path_name_data = config.paths.input_path + os.sep + config.dataset_config.data_dir + os.sep + mode
        self.sample_names = os.listdir(self.path_name_data)

        self.frame_transform = frame_transform
        self.video_transform = video_transform

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):

        sample_name = self.sample_names[idx]

        # Get metadata 
        speed_time = np.load(self.path_name_data + os.sep + str(sample_name) + os.sep + 'processed_log' + os.sep + 'CAN' + os.sep + 'speed' + os.sep + 't')
        speed_value = np.load(self.path_name_data + os.sep + str(sample_name) +  os.sep + 'processed_log' + os.sep + 'CAN' + os.sep + 'speed' + os.sep + 'value')

        # Get video object
        frame_reader = FrameReader(self.path_name_data + os.sep + sample_name + os.sep + 'video.hevc')
        video_frames = []
        for idx in range(len(speed_value)):
            frame = np.array(frame_reader.get(idx, pix_fmt='rgb24')[0], dtype=np.float64)
            frame = self.frame_transform(frame)
            video_frames.append(frame)
        video = torch.stack(video_frames, 0)

        if self.video_transform:
            video = self.video_transform(video)

        output = {
            'path': self.path_name_data + os.sep + str(sample_name),
            'video_frames': video,
            'speed_time': speed_time,
            'speed_value': speed_value
        }

        return output