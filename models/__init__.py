from models.spade_model import SpadeModel

def build_model(cfg):

    if cfg['TRAINING']['MODEL'] == 'init':
        model = SpadeModel(cfg)

    return model