
from src.models.cnn_lstm import CNNLSTM
from src.models.bandari_baseline import BandariBaseline

def build_model(config):

    if config.dataloader.dataloader_name == 'default':
        dataloader = BandariBaseline(config)
    else:
        raise NotImplementedError

    return dataloader

