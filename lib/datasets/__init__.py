# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from .aflw import AFLW
from .cofw import COFW
from .face300w import Face300W
from .wflw import WFLW

__all__ = ['AFLW', 'COFW', 'Face300W', 'WFLW', 'get_dataset']


def get_dataset(config):

    if config.DATASET.DATASET == 'AFLW':
        return AFLW
    elif config.DATASET.DATASET == 'COFW':
        return COFW
    elif config.DATASET.DATASET == '300W':
        return Face300W
    elif config.DATASET.DATASET == 'WFLW':
        return WFLW
    else:
        raise NotImplemented()


def get_preprocess(config):

    if config.DATASET.DATASET == 'AFLW':
        return AFLW(config, is_train=False)
    elif config.DATASET.DATASET == 'COFW':
        return COFW(config, is_train=False)
    elif config.DATASET.DATASET == '300W':
        return Face300W(config, is_train=False)
    elif config.DATASET.DATASET == 'WFLW':
        return WFLW(config, is_train=False)
    else:
        raise NotImplemented()

