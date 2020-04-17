import numpy as np
from scipy.signal import spectrogram
from pyrocko import gf

def normalize(y):
    y = y-np.mean(y)
    return y/abs(np.max(y))

def create_seismogram(source, target, engine, trim_length = 1000):
    trace = engine.process(source, [target]).pyrocko_traces()[0]
    y = trace.ydata  
    y = normalize(y) 
    return y[:trim_length]

def create_noisy_seismogram(source, target, engine, noise_factor=0.1, trim_length = 1000):
    trace = engine.process(source, [target]).pyrocko_traces()[0]
    y = trace.ydata
    noise = noise_factor * np.mean(abs(y))*np.random.randn(len(trace.ydata))
    y = noise + y                           
    y = normalize(y) #normalize the trace
    return y[:trim_length]

def create_spectrogram(seismogram, df):
    _, _, Sxx = spectrogram(seismogram, nperseg=df*2, scaling='spectrum')
    return Sxx/np.max(Sxx) #normalize

def trim_seismograms(seismograms):
    # trims seismograms to all be the length of the max length 
    trim = np.min([s.size for s in seismograms])
    return [s[:trim] for s in seismograms]


def createSynthetics(moment_tensors, stfs, depths, target, engine, noisy=False, noise_factor=0.1):
    
    '''
    createSynthetics: Create synthetic seismograms for the seimic spectrogram clustering experiment
    
    :param moment_tensors: A list of pyrocko.moment_tensor.MomentTensor objects
    :param stfs: List of pyrocko SourceTimeFunction objects.
    :param depths: List of floats, the depths for each seismogram
    :param target: A pyrocko.gf.Target object with information on the receiver
    :param engine: A pyrocko.gf.LocalEngine object initialized with the Green's functions store
    :param noisy: Boolean, whether to add Gaussian noise to the resulting synthetic 
    :param noise_factor: Float, controls the relative strength of the noise to the signal.  This constant multiplied by the mean of the absolute value of the spectrogram scales to additive noise.  Formula is :math `noise_factor * mean(|seismogram|) * seismogram`
    :returns A list containing arrays of synthetic seismograms and corresponding spectrograms -> [seismograms, spectrograms]
    
    '''

    assert len(moment_tensors)==len(stfs)==len(depths), print('moment_tensors, stfs, and depths must all be same length')
    
    seismograms = []
    
    for i, mt in enumerate(moment_tensors):
        source = gf.MTSource(lat=0, lon=0, depth=depths[i],
            mnn=mt.mnn, mee=mt.mee, mdd=mt.mdd,
            mne=mt.mne, mnd=mt.mne, med=mt.med,
            stf=stfs[i])
        if noisy: seismograms.append(create_noisy_seismogram(source, target, engine, noise_factor))
        else: seismograms.append(create_noisy_seismogram(source, target, engine))
    
    seismograms = trim_seismograms(seismograms)
    df = 10
    #deltat_inverse = 1/seismogram[0].deltat

    spectrograms = [create_spectrogram(s, df) for s in seismograms]

    spectrograms = np.array(spectrograms)
    
    specshape = spectrograms.shape
    #spectrograms = spectrograms.reshape(specshape[2],specshape[0],specshape[1])
    seismograms = np.array(seismograms)
    
    return[seismograms, spectrograms]
                
    
    