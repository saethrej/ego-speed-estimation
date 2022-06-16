"""

This script can be used to download MiDaS or check if it already exists
For further information see the README file

"""

import torch
import os

def main():
    if os.path.isdir("./torch_hub"):
        torch.hub.set_dir("./torch_hub")
        midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        print("Successfully loaded the models from torch_hub directory")
    else:
        midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        print("Succesfully loaded the models")
main()