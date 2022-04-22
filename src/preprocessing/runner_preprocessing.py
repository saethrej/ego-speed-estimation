from src.data_loader import default

def train_preprocessing(config):

    if config.preprocessing == 'default':
        data_transforms = default.train_preprocessing(config)
    else:
        raise NotImplementedError

    return data_transforms


def test_preprocessing(config):

    if config.preprocessing == 'default':
        data_transforms = default.test_preprocessing(config)
    else:
        raise NotImplementedError

    return data_transforms