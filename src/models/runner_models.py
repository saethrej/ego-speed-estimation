
from src.models.cnn_lstm import CNNLSTM
from src.models.bandari_baseline import BandariBaseline
from src.models.dual_cnn_lstm import DualCnnLstm
from src.models.dummy_model import DummyModel
from src.models.optical_flow_dummy import OpticalFlowDummy
from src.models.dof_cnn import DOFCNN

def build_model(config):

    if config.model.model_name == 'default':
        model = BandariBaseline(config)
    elif config.model.model_name == 'dual-cnn-lstm':
        model = DualCnnLstm(config)
    elif config.model.model_name == 'dummy':
        model = DummyModel(config)
    elif config.model.model_name == 'opticalflow':
        model = OpticalFlowDummy(config)
    elif config.model.model_name == 'depth_opticalflow':
        model = DOFCNN(config)
    else:
        raise NotImplementedError

    return model

