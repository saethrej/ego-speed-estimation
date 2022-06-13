"""

This file loads the model that was specified in the configuration

"""

import logging as log

from src.models.bandari_baseline import BandariBaseline
from src.models.dual_cnn_lstm import DualCnnLstm
from src.models.dof_cnn import DOFCNN
from src.models.dof_cnn_lstm import DOFCNNLSTM

def build_model(config):

    if config.model.model_name == 'bandari':
        log.info("Loading Model: BandaraBaseline")
        model = BandariBaseline(config)
    elif config.model.model_name == 'dual_cnn_lstm':
        log.info("Loading Model: DualCnnLstm")
        model = DualCnnLstm(config)
    elif config.model.model_name == 'dof_cnn':
        log.info("Loading Model: DOFCNN")
        model = DOFCNN(config)
    elif config.model.model_name == 'dof_cnn_lstm':
        log.info("Loading Model: DOFCNNLSTM")
        model = DOFCNNLSTM(config)
    else:
        log.info("Selected invalid Model. Aborting")
        raise NotImplementedError

    return model

