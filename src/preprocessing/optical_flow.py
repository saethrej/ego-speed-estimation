"""

This file implements the different preprocessing / frame annotations:
- Optical flow only (OpticalFlow)
- Optical flow and grayscale (OpticalFlowGrayScale)
- Optical flow, depth estimation and grayscale (OpticalFlowGrayScale) 


"""

import cv2 as cv
import numpy as np
import torch
import os

import logging as log

class OpticalFlow():
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # returns 2 channels: angle and magnitude
    def process(self, frames):
        np_frames = frames.numpy()

        # Optical Flow
        new_frames = np.empty((frames.shape[0], frames.shape[1], frames.shape[2], 2))
        prvs = cv.cvtColor(np_frames[0], cv.COLOR_BGR2GRAY)
        for i in range(1, frames.shape[0]):
            next = cv.cvtColor(np_frames[i], cv.COLOR_BGR2GRAY)
            new_frames[i] = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prvs = next
        new_frames[0] = new_frames[1] # copy value of second frame as first frame has no predecessor
        opt_flow = torch.tensor(new_frames).float().to(self.device)

        return opt_flow

class OpticalFlowGrayScale():
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # returns 3 channels: angle and magnitude of optical flow and grayscaled image
    def process(self, frames):
        np_frames = frames.numpy()

        # Optical Flow
        new_frames = np.empty((frames.shape[0], frames.shape[1], frames.shape[2], 2))
        gray_imgs = np.empty((frames.shape[0], frames.shape[1], frames.shape[2]))
        prvs = cv.cvtColor(np_frames[0], cv.COLOR_BGR2GRAY)
        gray_imgs[0] = prvs
        for i in range(1, frames.shape[0]):
            next = cv.cvtColor(np_frames[i], cv.COLOR_BGR2GRAY)
            gray_imgs[i] = next
            new_frames[i] = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prvs = next
        new_frames[0] = new_frames[1] # copy value of second frame as first frame has no predecessor
        opt_flow = torch.tensor(new_frames).float().to(self.device)
        gray_imgs = torch.tensor(gray_imgs).float().to(self.device)
        return_frames = torch.cat((opt_flow, gray_imgs.unsqueeze(dim=3)), dim=3)

        return return_frames

class OpticalFlowDepth():
    def __init__(self):
        '''
        # Note: For depth estimation we use MiDaS v2.1 small (https://github.com/isl-org/MiDaS)
        # This is loaded using torch.hub. If the machine on which this model is executed has no access
        # to the entire, the required contents need to be manually copied into a new directory 'torch_hub'
        # Further explanation can be found in the README.
        '''
        if os.path.isdir("./torch_hub"):
            # Load model from local directory if exists
            torch.hub.set_dir("./torch_hub")
        # Switch the following two lines to use a larger model (higher precision but slower)
        # self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        # Switch the following two lines to use a larger model (higher precision but slower)
        # self.transform = midas_transforms.dpt_transform
        self.transform = midas_transforms.small_transform

    # returns 4 channels: angles and magnitude of optical flow, depth estimation and grayscaled image
    def process(self, frames):
        np_frames = frames.numpy()

        # Optical Flow
        new_frames = np.empty((frames.shape[0], frames.shape[1], frames.shape[2], 2))
        gray_imgs = np.empty((frames.shape[0], frames.shape[1], frames.shape[2]))
        prvs = cv.cvtColor(np_frames[0], cv.COLOR_BGR2GRAY)
        gray_imgs[0] = prvs
        for i in range(1, frames.shape[0]):
            next = cv.cvtColor(np_frames[i], cv.COLOR_BGR2GRAY)
            gray_imgs[i] = next
            new_frames[i] = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prvs = next
        new_frames[0] = new_frames[1] # copy value of second frame as first frame has no predecessor
        opt_flow = torch.tensor(new_frames).float().to(self.device)
        gray_imgs = torch.tensor(gray_imgs).float().to(self.device)

        # Depth Estimation
        input_batch = torch.stack(
            [self.transform(np_frames[i]).squeeze(0) for i in range(np_frames.shape[0])],
            dim = 0
        ).to(self.device)        

        with torch.no_grad():
            depth_prediction = self.midas(input_batch)
            
            depth_prediction = torch.nn.functional.interpolate(
                depth_prediction.unsqueeze(0),
                size=np_frames.shape[1:3],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        # Put both together
        opt_depth = torch.cat((opt_flow, depth_prediction.unsqueeze(dim=3)), dim=3)
        return_frames = torch.cat((opt_depth, gray_imgs.unsqueeze(dim=3)), dim=3)

        return return_frames