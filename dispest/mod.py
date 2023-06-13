"""
This module contains functions for rf and demodulated datasets

Author: Wren Wightman (wren.wightman@duke.edu)
"""
import logging
import numpy as np
import scipy.signal as sig
logger = logging.getLogger(__name__)

def demodulate(rf:np.ndarray, fs:float, fd:float, fpass:float, fstop = None, usf:int=1, dsf:int=1, axis:int=0, t0:float=0, uswindow='tukey'):
    """Demodulate a given rf dataset by fd
    
    Parameters:
    ----
    `rf`: numpy array do be demodulated along `axis`
    `fs : sampling frequency along `axis` [Hz]
    `fd`: demodulation frequency [Hz]
    `fc`: Cuttoff frequency of filter. Default is `fd` [Hz]
    `usf`: upsample factor along `axis`
    `dsf`: downsample factor along `axis` applied after upsampling
    `axis`: axis on which to demodulate
    `uswindow`: frequency domain window type used in upsampling process

    Returns
    ----
    `i`: real part of demodulated signal
    `q`: imaginary part of demodulated signal
    `fs_new`: new sampling frequency = `usf`*`fs`/`dsf`
    """

    logger.info("Demodulating...")
    if usf != 1:
        logger.info(f"Upsampling dataset by a factor of {usf}")
        rf = sig.resample(x=rf, num=usf*rf.shape[axis], window=uswindow, domain='time')
        fs = fs*usf
    else:
        logger.info(f"No upsampling appplied")

    logger.info("Generating demodulation time vectors")
    t_shape = np.ones(len(rf.shape), dtype=int)
    t_shape[axis] = -1
    t = t0 + np.arange(rf.shape[axis])/fs
    t = t.reshape(t_shape)
    i_unfiltered = 2*np.cos(2*np.pi*fd*t) * rf
    q_unfiltered = 2*np.sin(2*np.pi*fd*t) * rf

    logger.info("Low-pass filtering...")
    if fstop is None:
        fstop = 1.5 * fpass
        logger.info(f"No fstop given, filling fstop with {fstop} Hz")
    N, Wn = sig.buttord(wp=2*fpass/fs, ws=2*fstop/fs, gpass=6, gstop=40)
    logger.info(f"Using an order {N} and {2*fs*Wn} Hz filter")
    b, a = sig.butter(N, Wn, btype="low", output='ba')
    i_filt = sig.filtfilt(b, a, i_unfiltered, axis=axis)
    q_filt = sig.filtfilt(b, a, q_unfiltered, axis=axis)
    del i_unfiltered, q_unfiltered

    if dsf != 1:
        logger.info(f"Downsampling by {dsf}")
        ds_slice = []
        for ind, n in enumerate(i_filt.shape):
            if ind == axis:
                ds_slice.append(slice(0, n, dsf))
            else:
                ds_slice.append(slice(n))
        ds_slice = tuple(ds_slice)
        i_filt = i_filt[ds_slice]
        q_filt = q_filt[ds_slice]
        fs = fs/dsf
    
    return i_filt, q_filt, fs