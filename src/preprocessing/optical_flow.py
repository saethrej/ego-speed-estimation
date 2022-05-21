import cv2 as cv
import numpy as np
import torch
import logging as log

def train_optical_flow(frames):
    np_frames = frames.numpy()
    new_frames = np.empty((frames.shape[0], frames.shape[1], frames.shape[2], 2))
    prvs = cv.cvtColor(np_frames[0], cv.COLOR_BGR2GRAY)
    for i in range(1, frames.shape[0]):
        next = cv.cvtColor(np_frames[i], cv.COLOR_BGR2GRAY)
        new_frames[i] = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prvs = next
    new_frames[0] = new_frames[1] # just copy first frame
    output = torch.tensor(new_frames)
    return output.float()