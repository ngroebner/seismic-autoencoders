import numpy as np
from pyrocko import gf
from pyrocko.gf import LocalEngine
from pyrocko import moment_tensor as mtm
from scipy.signal import spectrogram

from momentTensors import *
from synthetics import *

import pandas as pd

import os

from sacred.observers import FileStorageObserver

from sacred import Experiment
ex = Experiment('create synthetic seiosmograms for various focal mechanisms')
ex_dir = 'focalmechanism_synthetic_seismograms/'
ex.observers.append(FileStorageObserver(ex_dir))

    
def createFocalMechanisms(n, depths, durations, target, engine, noisy=False, noise_factor=0.1):
    n_0 = int(n/3)
    
    seismograms = []
    
    moment_tensors = [createMT_DC(1) for i in range(n_0)] + \
              [createMT_CLVD(1) for i in range(n_0)] + \
              [createMT_Isotropic(1) for i in range(n_0)]
            
    focal_mechanisms = n_0*['DC'] + n_0*['CLVD'] + n_0*['Isotropic']
    
    for i, mt in enumerate(moment_tensors):
        source = gf.MTSource(lat=0, lon=0, depth=depths[i],
            mnn=mt.mnn, mee=mt.mee, mdd=mt.mdd,
            mne=mt.mne, mnd=mt.mne, med=mt.med,
            stf=gf.BoxcarSTF(duration=durations[i]))
        if noisy: seismograms.append(create_noisy_seismogram(source, target, engine, noise_factor))
        else: seismograms.append(create_noisy_seismogram(source, target, engine))
            
    #seismograms = np.array(quakes) 
    df = 20
    #deltat_inverse = 1/seismogram[0].deltat
    
    spectrograms = [create_spectrogram(s, df) for s in seismograms]
    #spectrograms = np.array(spectrograms)
    
    #specshape = spectrograms.shape
    #spectrograms = spectrograms.reshape(specshape[2],specshape[0],specshape[1])

    
    df = pd.DataFrame({'FocalMechanism':focal_mechanisms,
                       'MomentTensor':moment_tensors,
                       'Seismogram':seismograms,
                       'Spectrogram':spectrograms,
                       'Depth':depths,
                       'Duration':durations})
    
    return df

@ex.named_config
def nonoise1():
    velocity_model = 'crust2_m5_hardtop_16Hz'
    n = 9000
    long = 0.5
    tmin = 10
    tmax = 75
    depths = np.random.choice([10000, 20000, 35000], size=n)
    durations = np.random.randint(1,10,size=n)
    noisy = False
    noise_factor = None
    save_dir = 'focalmechanism_synthetic_seismograms/nonoise/'
    save_file = 'synthetics.pkl'
    
@ex.named_config
def noisy1():
    velocity_model = 'crust2_m5_hardtop_16Hz'
    n = 9000
    long = 0.5
    tmin = 10
    tmax = 75
    depths = np.random.choice([10000, 20000, 35000], size=n)
    durations = np.random.randint(1,10,size=n)
    noisy = True
    noise_factor = 0.1
    save_dir = 'focalmechanism_synthetic_seismograms/noisy/'
    save_file = 'synthetics.pkl'

@ex.automain
def main(n, depths, durations, noisy, noise_factor, velocity_model, long, tmin, tmax, save_file, save_dir, _run):
    
    engine = LocalEngine(store_dirs=[velocity_model])

    target = gf.Target(
           quantity='displacement',
           lat=0, lon=long,
           store_id=velocity_model,
           codes=('NET', 'STA', 'LOC', 'E'),
           tmin=tmin, tmax=tmax)
    
    df = createFocalMechanisms(n, depths, durations, target, engine, noisy=noisy, noise_factor=noise_factor)
          
    if os.path.exists(save_dir)==False: os.mkdir(save_dir)
        
    path = save_dir + save_file

    df.to_pickle(path)
    
    ex.add_artifact(path)
    