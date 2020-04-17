# This file generates the synthetic seismograms needed for Experiment 1

import numpy as np
from pyrocko import gf
from pyrocko.gf import LocalEngine
from pyrocko import moment_tensor as mtm
from scipy.signal import spectrogram

from momentTensors import *
from synthetics import *

import pickle

import os


n_eachsource = 2000
n_seismograms = 3* n_eachsource

long = 0.5
tmin = 10
tmax = 20


save_dir = 'Experiment1'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
velocity_model = 'crust2_m5_hardtop_16Hz'

# create random moment tensors
print('Creating random moment tensors...')
moment_tensors = [createMT_DC(1) for i in range(n_eachsource)] + \
                 [createMT_CLVD(1) for i in range(n_eachsource)] + \
                 [createMT_Isotropic(1) for i in range(n_eachsource)]
 
#initialize source time functions to None - using impulses sources for this experiment
stfs = n_seismograms*[None]

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
    
'''np.save(seismograms, 'exp1_seismograms.npy')
np.save(spectrograms, 'exp1_spectrograms.npy')'''

fin = {'seismograms':seismograms,
       'spectrograms':spectrograms,
       'moment_tensors':[mt.m() for mt in moment_tensors], # dump the actual tensors
       'strike_dip_rakes':[mt.both_strike_dip_rake() for mt in moment_tensors],
       'source_mechanisms':n_eachsource*['DC'] + n_eachsource*['CLVD'] + n_eachsource*['Isotropic'],
       'depths':depths,
       'noise_types':n_seismograms*[None],
       'stfs':n_seismograms*[None],
       'durations':np.zeros(n_seismograms),
       'velocity_model':velocity_model}

with open(save_dir+"/synthetics.pkl","wb") as f:
    pickle.dump(fin,f)
