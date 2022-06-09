import logging as log

from src.models.cnn_lstm import CNNLSTM
from src.models.bandari_baseline import BandariBaseline
from src.models.dual_cnn_lstm import DualCnnLstm
from src.models.dummy_model import DummyModel
from src.models.optical_flow_dummy import OpticalFlowDummy
from src.models.dof_cnn import DOFCNN

def build_model(config):

    if config.model.model_name == 'default':
        log.info("Loading Model: BandaraBaseline")
        model = BandariBaseline(config)
    elif config.model.model_name == 'dual-cnn-lstm':
        log.info("Loading Model: DualCnnLstm")
        model = DualCnnLstm(config)
    elif config.model.model_name == 'dummy':
        log.info("Loading Model: DummyModel")
        model = DummyModel(config)
    elif config.model.model_name == 'opticalflow':
        log.info("Loading Model: OpticalFlowDummy")
        model = OpticalFlowDummy(config)
    elif config.model.model_name == 'depth_opticalflow':
        log.info("Loading Model: DOFCNN")
        model = DOFCNN(config)
    else:
        raise NotImplementedError

    return model

