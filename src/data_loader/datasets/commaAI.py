import logging as log
import os
import pickle
import math
from random import sample
import torch
import torchvision
import numpy as np


class CommaAI(torch.utils.data.Dataset):

    video_name = "video_comcro.mp4"
    speed_name = "speeds.npy"

    def __init__(self, config, mode, frame_transform, video_transform=None):
        super(CommaAI).__init__()

        # save the mode (train or val)
        self.mode = mode

        # save the transforms
        self.frame_transform = frame_transform
        self.video_transform = video_transform

        # store information on the number of frames of a video and a sample
        self.video_length = config.dataset_config.video_length
        self.sample_length = config.dataset_config.sample_length

        # create mapping from indices to paths of sample with that index
        self.sample_paths = None
        self.num_samples = 0
        self.__compute_mapping(config)


    def __len__(self):
        ''' returns the number of samples in the dataset'''
        return self.num_samples

    def __getitem__(self, idx):
        ''' returns a video sample with the given index '''

        video_path = os.path.join(self.sample_paths[idx], CommaAI.video_name)
        speed_path = os.path.join(self.sample_paths[idx], CommaAI.speed_name)

        # load video an extract a random subsequence of fixed length
        offset = np.random.randint(0, 48 - math.ceil(self.sample_length/25) - 1)
        video_frames = torchvision.io.read_video(
            video_path, 
            start_pts=offset, 
            end_pts=offset + self.sample_length/25 + 1, 
            pts_unit="sec"
        )[0][:self.sample_length].float()
        log.debug("Read video with idx {} and shape {}.".format(idx, video_frames.shape))

        # try to fix issue where video is not as long as expected
        if video_frames.shape[0] != self.sample_length:
            log.warning("Video read has shape {}, but required is len {}. Fetching new sample at beginning of video.".format(video_frames.shape, self.sample_length))
            offset = 0
            video_frames = torchvision.io.read_video(
                video_path, 
                start_pts=offset, 
                end_pts=offset + self.sample_length/25 + 1, 
                pts_unit="sec"
            )[0][:self.sample_length].float()
            log.info("Read video with idx {} again. New shape: {}".format(idx, video_frames.shape))
            

        # permute axis ([L,H,W,3] -> [L,3,H,W]) and apply transform to it (if applicable)
        video_frames = np.transpose(video_frames, (0, 3, 1, 2))
        if self.video_transform:
            video_frames = self.video_transform(video_frames)

        # load speed data and extract same subsequence
        frame_speeds = np.load(speed_path).flatten()[offset * 25 : offset * 25 + self.sample_length]
        log.debug("Tensor Size Speeds = {}".format(frame_speeds.shape))

        # return the frames and the speeds as a tuple
        return (video_frames, torch.from_numpy(frame_speeds).float())


    def __compute_mapping(self, config):
        ''' computes a mapping from index to the path to the corresponding sample'''
        skip = False
        if self.mode == "train":
            if os.path.exists(os.path.join(config.paths.input_path, config.dataset_config.mapping.train)):
                train_file = open(os.path.join(config.paths.input_path, config.dataset_config.mapping.train), "rb")
                self.sample_paths = pickle.load(train_file)['sample_paths']
                self.num_samples = len(self.sample_paths)
                skip = True
                train_file.close()

        elif self.mode == "val":
            if os.path.exists(os.path.join(config.paths.input_path, config.dataset_config.mapping.val)):
                val_file = open(os.path.join(config.paths.input_path, config.dataset_config.mapping.val), "rb")
                self.sample_paths = pickle.load(val_file)['sample_paths']
                self.num_samples = len(self.sample_paths)
                skip = True
                val_file.close()

        if not skip:
            log.info("Build mapping")
            if self.mode == "train":
                temp_dirs = [os.path.join(config.paths.input_path, config.dataset_config.dirs.root, d) for d in config.dataset_config.dirs.train]
            elif self.mode == "val":
                temp_dirs = [os.path.join(config.paths.input_path, config.dataset_config.dirs.root, d) for d in config.dataset_config.dirs.val]

            # walk through all training samples and record their paths
            sample_count = 0
            sample_list = []
            for dir in temp_dirs:
                for root, subfolders, files in os.walk(dir):
                    # skip all directories that do not contain a video and speed values
                    if not ("video.mp4" in files and "speeds.npy" in files):
                        continue

                    # add root to sample list and increase counter
                    sample_list.append(root)
                    sample_count += 1

            # sort the samples by their path for (perhaps) faster access
            sample_list.sort()
            self.sample_paths = sample_list
            self.num_samples = sample_count

            # dump to disk
            dict_to_dump = {"sample_paths": self.sample_paths}
            out_file = open(
                os.path.join(config.paths.input_path, config.dataset_config.mapping.train) if self.mode == "train" else os.path.join(config.paths.input_path, config.dataset_config.mapping.val),
                "wb"
            )
            pickle.dump(dict_to_dump, out_file)
            out_file.close()

        log.info("{} set has {} samples.".format(self.mode, self.num_samples))



