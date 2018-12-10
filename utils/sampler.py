from copy import copy
import math
import numpy as np

def sample_cfg(cfg):
    """
        Samples the config with max and min values using the given scale

        Parameter
        ---------
            - cfg: obj: argparse config
        Return
        ------
            - cfg_copy: obj: a copy of the config with a sample of the values
    """
    cfg_copy = copy(cfg)

    for attr in cfg_copy.__dict__.keys():
        attr_val = cfg_copy.__getattr__(attr)

        if (isinstance(attr_val, dict) and 'max' in attr_val):
            max_val = attr_val['max']
            min_val = attr_val['min']
            scale = attr_val['scale']

            new_val = None

            if scale == 'linear':
                new_val = np.random.uniform(low=min_val, high=max_val)
            elif scale == 'exponential':
                exponent = math.log(max_val, min_val)
                new_exponent = np.random.uniform(low=1, high=exponent)
                new_val = min_val ** new_exponent
            else:
                raise ValueError('Scale is not \'linear\' or \'exponential\'')
            cfg_copy.__setattr__(attr, new_val)

        if (isinstance(attr_val, list)):
            index = np.random.randint(0, len(attr_val))
            new_val = attr_val[index]
            cfg_copy.__setattr__(attr, new_val)

    return cfg_copy