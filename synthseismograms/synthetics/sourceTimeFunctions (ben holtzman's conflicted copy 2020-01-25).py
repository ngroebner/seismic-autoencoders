import numpy as np
from pyrocko import gf
from pyrocko import moment_tensor as mtm
from pyrocko.gf import STF
from pyrocko.guts import Float, Int
from pyrocko.gf.seismosizer import sshift
from pyrocko import trace, util, config, model
from scipy.stats import lognorm
import matplotlib.pyplot as plt

class LognormalSTF:
    
    def __init__(self, shape=1.5, scale=10, loc=0, duration=10, anchor=0, **kwargs):
        self.shape = shape
        self.scale = duration
        self.loc = loc
        self.duration = duration
        self.anchor = anchor      
        super().__init__(**kwargs)
          
    def _lognorm(self, nt):  
        x = lognorm.pdf(nt, s=self.shape, scale=self.scale, loc=self.loc)
        return self.duration * x

    def discretize_t(self, deltat, tref):
        # method returns discrete times and the respective amplitudes
        tmin_stf = tref - self.duration * (self.anchor + 1.) * 0.5
        tmax_stf = tref + self.duration * (1. - self.anchor) * 0.5
        tmin = round(tmin_stf / deltat) * deltat
        tmax = round(tmax_stf / deltat) * deltat
        nt = int(round((tmax - tmin) / deltat)) + 1
        times = np.linspace(tmin, tmax, nt)
        newtimes = np.linspace(0, tmax-tmin, nt)
        amplitudes = self._lognorm(newtimes)
        
        return times, amplitudes
    
    
class NoisyBoxcar:
    
    def __init__(self, noise_func, duration=1, anchor=0):
        self.noise_func = noise_func
        self.duration = duration
        self.anchor = anchor
    
    @classmethod
    def factor_duration_to_effective(cls):
        return 1.0

    def centroid_time(self, tref):
        return tref - 0.5 * self.duration * self.anchor
    
    def discretize_t(self, deltat, tref):
        tmin_stf = tref - self.duration * (self.anchor + 1.) * 0.5
        tmax_stf = tref + self.duration * (1. - self.anchor) * 0.5
        tmin = np.round(tmin_stf / deltat) * deltat
        tmax = np.round(tmax_stf / deltat) * deltat
        nt = int(np.round((tmax - tmin) / deltat)) + 1
        times = np.linspace(tmin, tmax, nt)
        amplitudes = np.ones_like(times)
        if times.size > 1:
            t_edges = np.linspace(
                tmin - 0.5 * deltat, tmax + 0.5 * deltat, nt + 1)
            t = tmin_stf + self.duration * np.array(
                [0.0, 0.0, 1.0, 1.0], dtype=np.float)
            f = np.array([0., 1., 1., 0.], dtype=np.float)
            amplitudes = util.plf_integrate_piecewise(t_edges, t, f)
            amplitudes /= np.sum(amplitudes)

        amplitudes *= self.noise_func(nt)
        tshift = (np.sum(amplitudes * times) - self.centroid_time(tref))

        return sshift(times, amplitudes, -tshift, deltat)

class NoisySTF(STF):
    #duration, noise, envelope
    
    def __init__(self, envelope_func, noise_func, duration, anchor, **kwargs):
        self.duration = duration
        self.anchor = anchor
        self.envelope_fnc = envelope_func
        self.noise_fnc = noise_func
        super().__init__(**kwargs)
    
    def discretize_t(self, deltat, tref):
        # method returns discrete times and the respective amplitudes
        tmin_stf = tref - self.duration * (self.anchor + 1.) * 0.5
        tmax_stf = tref + self.duration * (1. - self.anchor) * 0.5
        tmin = round(tmin_stf / deltat) * deltat
        tmax = round(tmax_stf / deltat) * deltat
        nt = int(round((tmax - tmin) / deltat)) + 1
        times = np.linspace(tmin, tmax, nt)
        noise = self.noise_fnc(nt)
        env = self.envelope_fnc(times)
        amplitudes = noise * env
        return times, amplitudes
    
    def plot_example(self, deltat, tref):
        # plots an example envelope and noise functions
        tmin_stf = tref - self.duration * (self.anchor + 1.) * 0.5
        tmax_stf = tref + self.duration * (1. - self.anchor) * 0.5
        tmin = round(tmin_stf / deltat) * deltat
        tmax = round(tmax_stf / deltat) * deltat
        nt = int(round((tmax - tmin) / deltat)) + 1
        times = np.linspace(tmin, tmax, nt)
        noise = self.noise_fnc(nt)
        env = self.envelope_fnc(times)
        
        plt.subplot(211)
        plt.scatter(times,noise)
        plt.title("Example noise")
        print(env)
        plt.subplot(212)
        plt.scatter(times, env)
        plt.title('Example envelope')


# Factory functions for noise and envelope 
def GRNoise(a=1, b=1.2, mag=1):
    
    def _cdf_inv(nt):
        x = np.random.uniform(0,mag,nt)
        return 10**((1/b)*(np.log10(1/x) - a))
    
    return _cdf_inv


def gaussianNoise(mu=0, sigma=1):
    
    def _random_sample(nt):
        x = np.random.uniform(0,1,nt)
        return np.abs(np.random.normal(x))
    
    return _random_sample
    
    
def lognormalEnvelope(shape, scale=1, loc=0):
    
    def _lognorm_envelope(nt):  
        return lognorm.pdf(nt, s=shape, scale=scale, loc=loc)
    
    return _lognorm_envelope

def createLognormalGRNoise(duration,a,b,shape=2,mag=1):
    
    env_tmp = lognormalEnvelope(shape=shape, scale=duration)
    def envelope(nt):
        return duration*env_tmp(nt)
    
    noise = GRNoise(a=a,b=b,mag=mag)
    
    return NoisySTF(envelope, noise, duration, anchor=0)


def createLognormalGaussianNoise(duration,mu=0,sigma=1,shape=2):
    # in order to make the shape and peak amplitude of this function invariant with the duration, 
    # need to set the scale equal to the duration (to make shape the same), 
    # and then multiply the whole function by the duration (to keep the peak the same)
    env_tmp = lognormalEnvelope(shape=shape, scale=duration)
    def envelope(nt):
        return duration*env_tmp(nt)
    
    noise = gaussianNoise(mu=mu, sigma=sigma)
    
    return NoisySTF(envelope, noise, duration, anchor=0)