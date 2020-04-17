# This file generates the synthetic seismograms needed for Experiment 1

'''Experiment 2. Ability for AEC to determine the source time function class
Lognormal, boxcar, and triangular source time functions will be used to generate seismograms with a single source mechanism type (double couple), random nodal plane orientation, and varying durations (1 - 10 seconds).
'''

import numpy as np
from pyrocko import gf
from pyrocko.gf import LocalEngine
from pyrocko import moment_tensor as mtm
from scipy.signal import spectrogram

from momentTensors import *
from synthetics import *
from sourceTimeFunctions import LognormalSTF

import pickle

import os


n_eachstf = 2000
n_seismograms = 3* n_eachstf

long = 0.5
tmin = 10
tmax = 30


save_dir = 'Experiment2'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
velocity_model = 'crust2_m5_hardtop_16Hz'

# create random moment tensors
print('Creating random moment tensors...')
moment_tensors = [createMT_DC(1) for i in range(n_seismograms)] 
source_mechanisms = n_seismograms*['DC']
 
#initialize source time functions 
durations = np.random.randint(1,11,n_eachstf)
stfs = [LognormalSTF(shape=2, duration=dur) for dur in durations] + \
       [gf.BoxcarSTF(duration=dur) for dur in durations] + \
       [gf.TriangularSTF(duration=dur) for dur in durations]

stftypes = n_eachstf*['lognorm'] + n_eachstf*['boxcar'] +n_eachstf*['triangle']

# uniform depth of 20 km / 20,000m
depths = n_seismograms*[20000] 

# seismometer target and the actual computation engine
target = gf.Target(
           quantity='displacement',
           lat=0, lon=long,
           store_id=velocity_model,
           codes=('NET', 'STA', 'LOC', 'E'),
           tmin=tmin, tmax=tmax)
engine = LocalEngine(store_dirs=[velocity_model])

# all of that for this!
print('Creating synthetic seismograms...')
seismograms, spectrograms = createSynthetics(moment_tensors, stfs, depths, target, engine)
    
fin = {'seismograms':seismograms,
       'spectrograms':spectrograms,
       'moment_tensors':[mt.m() for mt in moment_tensors], # dump the actual tensors,
       'strike_dip_rakes':[mt.both_strike_dip_rake() for mt in moment_tensors],
       'source_mechanisms':source_mechanisms,
       'depths':depths,
       'noise_types':n_seismograms*[None],
       'stfs':stftypes,
       'durations':3*durations.tolist(),
       'velocity_model':velocity_model}

with open(save_dir+"/synthetics.pkl","wb") as f:
    pickle.dump(fin,f)
