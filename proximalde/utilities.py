import numpy as np


def _check_input(*args):
    ''' Ensures that all matrices are 2d and returns
    2d variant of any column vector as a (n, 1) array
    '''
    # check that all variables have the same samples
    if len(np.unique([arg.shape[0] for arg in args])) > 1:
        raise AttributeError('All input variables need to have the same number '
                             'of samples')
    # reshape all variables to be 2d matrices
    return (arg.reshape(-1, 1) if len(arg.shape) == 1 else arg for arg in args)
