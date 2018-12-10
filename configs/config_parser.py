import yaml
import os

CONFIG_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(CONFIG_DIR)
IDX_DIR = os.path.join(ROOT_DIR, 'phn_index_maps')

DATA_TYPE_DIM_MAP = {
    'mfcc': 13,
    'mfcc-delta': 13,
    'ssc': 13,
    'ssc-delta': 13,
    'logfbank': 26,
    'logfbank-delta': 26,
    'logfbank_40': 40,
    'logfbank_40-delta': 40
}

class ConfigObjFromDict(object):
    """ Handy class for creating an object updated as a dict but accessed as an obj. """
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        return self.__dict__.get(name, None)

    def __str__(self):
        return ' '.join(['{0}: {1}\n'.format(k, v) for k, v in self.__dict__.items()])

    def __copy__(self):
        return ConfigObjFromDict(**self.__dict__)

    def to_dict(self):
        return self.__dict__

def parse(args):
    """ Takes in the name of the config file and the parser.args and return an object with all the config """
    config_filename = args.config + '.yaml'
    config_path = os.path.join(CONFIG_DIR, config_filename)
    if not os.path.exists(config_path): raise ValueError("Config: {} does not exist".format(config_path))

    # load yaml config file
    with open(config_path, 'r') as config_file:
        config_dict = yaml.load(config_file)

    for k, v in vars(args).items():
        if v:
            config_dict[k] = v

    cfg = ConfigObjFromDict(**config_dict)

    # load phn to index mapping
    if not hasattr(cfg, 'phn_idx_map_filename'): raise ValueError('phn_idx_map_filename is not in the config')
    phn_idx_path = os.path.join(IDX_DIR, cfg.phn_idx_map_filename)
    if not os.path.exists(phn_idx_path): raise ValueError("Phn Index Map: {} does not exist".format(phn_idx_path))

    with open(phn_idx_path, 'r') as phn_idx_file:
        cfg.phn_idx_map = yaml.load(phn_idx_file)

    cfg.num_classes = len(set(v for _, v in cfg.phn_idx_map.items()))
    cfg.num_channels = 3 if (cfg.data_type.endswith('delta')) else 1

    cfg.num_dimensions = DATA_TYPE_DIM_MAP[cfg.data_type]

    return cfg