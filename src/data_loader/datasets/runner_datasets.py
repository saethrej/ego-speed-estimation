"""

This file loads the dataset that was specified in the configuration

"""

from src.data_loader.datasets.commaAI import CommaAI


def get_video_dataset(config, mode, frame_transfroms, video_transforms=None):

    if config.dataloader.dataset_name == "comma-ai":
        return CommaAI(config=config, mode=mode, frame_transform=frame_transfroms, video_transform=video_transforms)
    else:
        raise NotImplementedError