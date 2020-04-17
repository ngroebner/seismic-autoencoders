import numpy as np
from pyrocko import gf
from pyrocko.gf import LocalEngine
from pyrocko import moment_tensor as mtm
from scipy.signal import spectrogram
from scipy.stats import possion

import os

def randomMT(mttype, mag):
    '''
    Returns a randomly rotated moment tensor 
    '''
    # must be one of double couple, continuous linear vector dipole, isoptropic sources
    mttypes = ['DC', 'CLVD', 'Isotropic']
    assert mttype in mttypes, print('mttype must be one of {}'.format(mttypes))

    if mttype == 'DC':
        return mtm.MomentTensor.random_dc(magnitude=mag)
    elif mttype =='CLVD':
        tensor = np.array([[1,0,0],[0,1,0],[0,0,-2]])
    elif mttype == 'Isotropic':
        tensor = np.array([[1,0,0],[0,1,0],[0,0,1]])

    rotated_tensor =  mtm.random_rotation() * tensor

    return mtm.MomentTensor(rotated_tensor, magnitude=mag)

from pyrocko.gf import STF
from pyrocko.guts import Float, Int


# maybe change this to a Poisson distribution source time function, and have either GR or gaussian noise
class GRLawSTF(STF):
    '''Poisson distributed random impulses based on the Gutenberg-Richter law.'''

    duration = Float.T(
        default=0.0,
        help='baseline of the ramp')

    anchor = Float.T(
        default=0.0,
        help='anchor point with respect to source-time: ('
             '-1.0: left -> source duration [0, T] ~ hypocenter time, '
             ' 0.0: center -> source duration [-T/2, T/2] ~ centroid time, '
             '+1.0: right -> source duration [-T, 0] ~ rupture end time)')

    mag = Float.T(
        default=5.0,
        help='magnitude used for Gutenberg-Richter law')

    b = Float.T(
        default=1.2,
        help='b parameter for Gutenberg-Richter law')

    poisson_envelope = Int.T(
        default = 0,
        help='Whether to apply a Poisson distribution envelope to the magnitudes of the asperities.')

    mu = Float.T(
        default = 10,
        help = 'Mu parameter for Poisson envelope')

    loc = Float.T(
        optional = True,
        help = "'loc' parameter for Poisson distribution function")


    def discretize_t(self, deltat, tref):
        # Gutenberg-Ricther law
        M = np.linspace(0,self.mag,1000)
        pdf = self.b*np.log(10)*10**(-self.b*M)
        self.cdf = 1-10**(-self.b*M)

        # method returns discrete times and the respective amplitudes
        tmin_stf = tref - self.duration * (self.anchor + 1.) * 0.5
        tmax_stf = tref + self.duration * (1. - self.anchor) * 0.5
        tmin = round(tmin_stf / deltat) * deltat
        tmax = round(tmax_stf / deltat) * deltat
        nt = int(round((tmax - tmin) / deltat)) + 1
        times = np.linspace(tmin, tmax, nt)
        if nt > 1:
            self._amplitudes = np.random.uniform(min(self.cdf),max(self.cdf),nt)
            if self.poisson_envelope:
                x = np.linspace(0,nt-1,nt)
                self.envelope = poisson.pmf(x, mu=self.mu, loc=self.loc)
                self.amplitudes = self.mag*(self._amplitudes * self.envelope)
            else:
                self.amplitudes = self._amplitudes
        else:
            self.amplitudes = np.ones(1)

        return times, self.amplitudes

    def base_key(self):
        # method returns STF name and the values
        return (self.__class__.__name__, self.duration, self.anchor)


if __name__ == '__main__':

    model = 'crust2_m5_hardtop_16Hz'

    engine = LocalEngine(store_dirs=[model])

    target = gf.Target(
           quantity='displacement',
           lat=0, lon=1,
           store_id=model,
           codes=('NET', 'STA', 'LOC', 'E'),
           tmin=10, tmax=75)


    DC_tensors = [createMT_DC(mag) for mag in 4*np.random.rand(1000)]
    CLVD_tensors = [createMT_CLVD(mag) for mag in 4*np.random.rand(1000)]
    Iso_tensors = [createMT_Isotropic(mag) for mag in 4*np.random.rand(1000)]

    moment_tensors = DC_tensors + CLVD_tensors + Iso_tensors

    def process(source, target):
        trace = engine.process(source, [target]).pyrocko_traces()[0]
        trace.ydata /= np.max(trace.ydata)
        return trace

    quakes_by_mechanism = []

    min_depth = 1000 # in meters
    max_depth = 35000
    depths = np.round((max_depth-min_depth)*(np.random.rand(len(moment_tensors)))+min_depth, 0) #depths from 1 km to 35 km

    min_duration = 1 # seconds
    max_duration = 10
    durations = np.random.randint(min_duration,max_duration+1,len(moment_tensors))

    # use a boxcar source-time function
    for i, mt in enumerate(moment_tensors):
        source = gf.MTSource(lat=0, lon=0, depth=depths[i],
            mnn=mt.mnn, mee=mt.mee, mdd=mt.mdd,
            mne=mt.mne, mnd=mt.mne, med=mt.med,
            stf=gf.BoxcarSTF(duration=durations[i]))
        quakes_by_mechanism.append(process(source, target))


    seismograms = np.array([q.ydata[:1000] for q in quakes_by_mechanism]) #trim to 1000 samples
    df = 20
    deltat_inverse = 1/quakes_by_mechanism[0].deltat


    spectrograms = [spectrogram(s, fs=deltat_inverse,nperseg=df*2, scaling='spectrum')[2] for s in seismograms]

    spectrograms=np.dstack(spectrograms)

    specshape = spectrograms.shape
    spectrograms = spectrograms.reshape(specshape[2],specshape[0],specshape[1])

    path='three_mechanisms_synthetic_seismograms/'

    if os.path.exists(path)==False: os.mkdir(path)

    np.save((path+'/depths.npy'), depths)
    np.save((path+'/durations.npy'), durations)
    np.save((path+'/seismograms.npy'), seismograms)
    np.save((path+'/spectrograms.npy'), spectrograms)

