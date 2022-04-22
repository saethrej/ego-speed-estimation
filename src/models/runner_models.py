
from src.models.cnn_lstm import CNNLSTM

def build_model(config):

    if config.dataloader.dataloader_name == 'default':
        dataloader = CNNLSTM(config)
    else:
        raise NotImplementedError

    return dataloader

