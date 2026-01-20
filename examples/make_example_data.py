import numpy as np
import scipy.signal as sig
from tqdm import tqdm
from timuscle.dataio import putdictasHDF5

#%% Make acoustic tracking data with an axial PSF
# define the imaging PSF
bw=80/100
f = 5E6
c = 1540
fs = 1E9

tc = sig.gausspulse('cutoff', bw=bw, tpr=-80, fc=f)
Ntc = np.ceil(tc*fs)
timp = (np.arange(2*Ntc+1)-Ntc)/fs
imp = sig.gausspulse(t=timp, bw=bw, tpr=-80, fc=f)

def sim_trace(xs, amps, imp, tstart, tstop, fs, c:float=1540, nbatch:int=512):
    import cupy as cp
    if (np.ndim(xs) != 1) or (np.ndim(amps) !=1) or (np.ndim(imp) !=1):
        raise ValueError("xs, amps, and imp must be 1D")
    if len(xs) != len(amps):
        raise ValueError("xs and amps must be the same length")
    
    nt = int((tstop-tstart)*fs)
    t = tstart + np.arange(nt)/fs

    if len(imp) > nt:
        raise ValueError("Time trace mut be bigger than the impulse response")   

    W = cp.array(np.exp(-2j*np.pi*np.arange(nt)/nt))

    indxs = cp.array(2*fs*xs/c)
    amps = cp.array(amps)
    nx = len(indxs)

    SCAT = 0
    tbar = tqdm(total=nx)
    for idx0 in range(0, nx, nbatch):
        idxn = np.min([nx, idx0+nbatch])
        SCAT += cp.sum(amps[idx0:idxn,None] * W[None,:] ** indxs[idx0:idxn,None], axis=0)
        tbar.update(idxn-idx0)
    tbar.close()

    SCAT = cp.asnumpy(SCAT)

    IMP = np.fft.fft(imp, nt) * cp.asnumpy(W) **(-len(imp)/2)
    TRACE = SCAT * IMP
    trace = np.real(np.fft.ifft(TRACE))
    return trace, t

# define the scatterer field
xmin = (c/f)/2
xmax = 10E-3 - (c/f)/2
Nx = int(1E7)

rng = np.random.default_rng(0)

x0 = rng.uniform(xmin, xmax, Nx)
a0 = rng.normal(0, 1, Nx)

# Undisplaced signal
trace0, t = sim_trace(x0, a0, imp, 0, 20E-3/c, fs, c)
x = c*t/2

# define the displacement field
f_disp = lambda x, mag=20E-6: mag*np.exp(-(x-5E-3)**2/(1E-3**2))

x1 = x0 + f_disp(x0)

# displaced signal
trace1, _ = sim_trace(x1, a0, imp, 0, 20E-3/c, fs, c)

#%% Save the simulated data
putdictasHDF5("example_data.h5", 
    dict(
        tracking_data = dict(
            t = t,
            traces = np.array([trace0, trace1]),
            disp0  = f_disp(x),
            fc = f,
            fs = fs,
            c = c
        )
    )
)