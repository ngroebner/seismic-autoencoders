# This file generates the synthetic seismograms needed for Experiment 3

'''Experiment 3. Ability for AEC to determine noise statistics of the source time function
A lognorm source time function with gaussian and gutenberg-richter noise will be used with a single source mechanism (DC) and random nodal planes'''

import numpy as np
from pyrocko import gf
from pyrocko.gf import LocalEngine
from pyrocko import moment_tensor as mtm
from scipy.signal import spectrogram

from momentTensors import *
from synthetics import *
from sourceTimeFunctions import createLognormalGRNoise, createLognormalGaussianNoise

import pickle

import os


n_eachstf = 2500
n_seismograms = 2 * n_eachstf

long = 0.5
tmin = 10
tmax = 30

# parameters for the source time functions
a = 1
b = 4

save_dir = 'Experiment3'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
velocity_model = 'crust2_m5_hardtop_16Hz'

# create random moment tensors
print('Creating random moment tensors...')
moment_tensors = [createMT_DC(1) for i in range(n_seismograms)] 
source_mechanisms = n_seismograms*['DC']
 
#initialize source time functions 
GRSTF = createLognormalGRNoise
durations = np.random.randint(1,11,n_eachstf)
stfs = [createLognormalGRNoise(duration=dur, a=a, b=b) for dur in durations] + \
       [createLognormalGaussianNoise(duration=dur) for dur in durations]

stftypes = n_eachstf*['lognorm'] + n_eachstf*['lognorm'] 
noise_types = n_eachstf*['GR'] + n_eachstf*['Gaussian']

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
       'moment_tensors':[mt.m() for mt in moment_tensors], # dump the actual tensors
       'strike_dip_rakes':[mt.both_strike_dip_rake() for mt in moment_tensors],
       'source_mechanisms':source_mechanisms,
       'depths':depths,
       'noise_types':noise_types,
       'stfs':stftypes,
       'durations':2*durations.tolist(),
       'velocity_model':velocity_model}

with open(save_dir+"/synthetics.pkl","wb") as f:
    pickle.dump(fin,f)

