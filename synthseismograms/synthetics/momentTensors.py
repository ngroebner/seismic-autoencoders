import numpy as np
from pyrocko import moment_tensor as mtm
from pyrocko import gf


def createMT_DC(mag):

    return mtm.MomentTensor.random_dc(magnitude=mag)

def createMT_CLVD(mag):
    exp = mtm.magnitude_to_moment(mag)

    # init pyrocko moment tensor
    cvld_matrix = np.array([[1,0,0],[0,1,0],[0,0,-2]]) * exp
    rotated_matrix = mtm.random_rotation() * cvld_matrix
    return mtm.MomentTensor(rotated_matrix)


# not needed - have gf.ExplosionSource to create
def createMT_Isotropic(mag):
    return gf.ExplosionSource().pyrocko_moment_tensor()

