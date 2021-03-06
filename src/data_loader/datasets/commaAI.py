"""

Loading of dataset

"""


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

    def __init__(self, config, mode, frame_transform, video_transform=None, device=None):
        super(CommaAI).__init__()

        # set device used
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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

        # load video and extract a random subsequence of fixed length
        offset = np.random.randint(0, 48 - math.ceil(self.sample_length/25) - 1)
        video_frames = torchvision.io.read_video(
            video_path, 
            start_pts=offset, 
            end_pts=offset + self.sample_length/25 + 1, 
            pts_unit="sec"
        )[0][:self.sample_length].float()
        log.debug("Read video with idx {} and shape {}.".format(idx, video_frames.shape))

        # Recover from issues during video loading, i.e. if shorter than expected or file is faulty
        if video_frames.shape[0] != self.sample_length:
            # Frames do not have desired shape. First try to fetch from beginning of this video
            log.debug("Video read has shape {}, but required is len {}. Fetching new sample at beginning of video.".format(video_frames.shape, self.sample_length))
            offset = 0
            video_frames = torchvision.io.read_video(
                video_path, 
                start_pts=offset, 
                end_pts=offset + self.sample_length/25 + 1, 
                pts_unit="sec"
            )[0][:self.sample_length].float()
            while (video_frames.shape[0] != self.sample_length):
                # If video frames are faulty or video is to short selected another index and sample from there until correct length
                if(idx > 0):
                    idx -= 1
                else:
                    idx = len(self.sample_paths) - 1
                log.debug("Video is faulty. Try other video with idx {}.".format(idx))
                video_path = os.path.join(self.sample_paths[idx], CommaAI.video_name)
                speed_path = os.path.join(self.sample_paths[idx], CommaAI.speed_name)

                offset = np.random.randint(0, 48 - math.ceil(self.sample_length/25) - 1)
                video_frames = torchvision.io.read_video(
                    video_path, 
                    start_pts=offset, 
                    end_pts=offset + self.sample_length/25 + 1, 
                    pts_unit="sec"
                )[0][:self.sample_length].float()
            
            log.debug("Read new video with idx {}. New shape: {}".format(idx, video_frames.shape))

        if self.video_transform:
            # apply transform to it and permute axis ([L,H,W,3] -> [L,3,H,W])
            video_frames = self.video_transform.process(video_frames)
            video_frames = torch.permute(video_frames, (0, 3, 1, 2))
        else:
            # permute axis ([L,H,W,3] -> [L,3,H,W])
            video_frames = np.transpose(video_frames, (0, 3, 1, 2))

        # load speed data and extract same subsequence
        frame_speeds = np.load(speed_path).flatten()[offset * 25 : offset * 25 + self.sample_length]
        frame_speeds = torch.from_numpy(frame_speeds).float()
        log.debug("Tensor Size Speeds = {}".format(frame_speeds.shape))

        # return the frames and the speeds as a tuple
        return (video_frames, frame_speeds)


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

        elif self.mode == "test":
            if config.dataset_config.day_only:
                this_path = config.dataset_config.mapping.test_day
            else:
                this_path = config.dataset_config.mapping.test
            if os.path.exists(os.path.join(config.paths.input_path, this_path)):
                test_file = open(os.path.join(config.paths.input_path, this_path), "rb")
                self.sample_paths = pickle.load(test_file)['sample_paths']
                self.num_samples = len(self.sample_paths)
                skip = True
                test_file.close()

        # Create mapping if not already existing or force argument set in config
        if not skip or config.dataset_config.mapping.rebuild_mapping:
            log.info("Build mapping")
            if self.mode == "train":
                temp_dirs = [os.path.join(config.paths.input_path, config.dataset_config.dirs.root, d) for d in config.dataset_config.dirs.train]
            elif self.mode == "val":
                temp_dirs = [os.path.join(config.paths.input_path, config.dataset_config.dirs.root, d) for d in config.dataset_config.dirs.val]
            elif self.mode == "test":
                temp_dirs = [os.path.join(config.paths.input_path, config.dataset_config.dirs.root, d) for d in config.dataset_config.dirs.test]
            # walk through all training samples and record their paths
            sample_count = 0
            sample_list = []
            for dir in temp_dirs:
                for root, subfolders, files in os.walk(dir):
                    # skip all directories that do not contain a video and speed values
                    if not (CommaAI.video_name in files and CommaAI.speed_name in files):
                        continue

                    if self.mode == "test" and config.dataset_config.day_only: 
                        hour = int(root.split("-")[-3])
                        if hour < 6 or hour > 20:
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
            if self.mode == "train":
                mapping_path = config.dataset_config.mapping.train
            elif self.mode == "val":
                mapping_path = config.dataset_config.mapping.val
            elif self.mode == "test":
                if config.dataset_config.day_only:
                    mapping_path = config.dataset_config.mapping.test_day
                else:
                    mapping_path = config.dataset_config.mapping.test
            out_file = open(
                os.path.join(config.paths.input_path, mapping_path),
                "wb"
            )
            pickle.dump(dict_to_dump, out_file)
            out_file.close()

        log.info("{} set has {} samples.".format(self.mode, self.num_samples))



