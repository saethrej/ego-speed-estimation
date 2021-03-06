"""

This script extracts the speed values at each video frame and stores it in speeds.npy

"""


import os
import math
from posixpath import dirname
import numpy as np

speedfile = "value"
timefile = "t"

def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")

    for i in range(len(idx)):
        if idx[i] > 0 and (idx[i] == len(array) or math.fabs(value[i] - array[idx[i]-1]) < math.fabs(value[i] - array[idx[i]])):
            idx[i] -= 1
    
    return idx


def extract():
    '''extracts the velocities at each frame'''
    
    # walk the data directory
    for root, subfolders, files in os.walk(os.path.join(os.getcwd(), "data/comma2k19")):
        # skip all directories that do not contain CAN speed values
        if not root.endswith('CAN/speed'):
            continue

        speeds = np.load(os.path.join(root, speedfile))
        times = np.load(os.path.join(root, timefile))

        # subtract start time from all time stamps 
        true_times = times - times[0]
        
        # segments are 1200 frames and 60s long -> 20 fps
        ideal_times = np.linspace(0, 59.95, 1200)

        # find the indices that store the time stamps closest to the ideal one
        indices = find_nearest(true_times.flatten(), ideal_times.flatten())
        extracted_speeds = speeds[indices]
        
        # dump extracted speeds to a file
        filename = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(root))), "speeds.npy")
        np.save(filename, extracted_speeds)

    return

if __name__ == "__main__":
    extract()
