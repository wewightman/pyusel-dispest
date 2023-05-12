"""
This module contains helpers for various beamforming functions

Author: Wren Wightman (wren.wightman@duke.edu)
"""
import logging
import numpy as np
import scipy.signal as sig
logger = logging.getLogger(__name__)

def demodulate(data, fd, fs, fc, tstart=0, axis=0, usf:int=1, dsf:int=1, filter='tukey'):
    """demodulates data, a numpy tensor, along the given 'axis'
    
    Paramaters:
        data: numpy tensor with data. Must contain 'axis'dimension
        fd: demodulation frequency (must be a single number or have the same length as data.shape[axis])
        fs: sampling frequency [Hz] along 'axis'
        fc: cuffoff frequency [Hz] to truncate the frequency domain at
        tstart: the time associated with the first sample along 'axis'
        axis: the axis along which to demodulate
        usf: upsample factor (integer)
        dsf: downsample factor (integer)
        filter: the type of low pass filter used
    Returns:
        I: the in-phase data
        Q: the quatrature/out of phase data
    """
    logger.info("Demodulating  via the _demodulate_numpy method...")

    if not usf == 1:
        data = sig.resample_poly(data, up=usf, down=1, axis=axis, window=filter)
        fs=usf*fs

    # check that axis is within bounds of data
    if np.ndim(data) <= axis:
        raise Exception(f"Axis is {axis}. Must be < {np.ndims(data)} for this data...")

    # make a reshaping vector for tn and fd (if fd is a vector)
    vreshape = np.ones(len(data.shape), dtype=int)
    vreshape[axis] = data.shape[axis]

    # validate that fd is either a single number or an array of length data.shape[axis]
    if np.ndim(fd)!=0:
        if len(fd) == 1:
            fd = float(fd)
        elif len(fd) == data.shape[axis]:
            fd = np.reshape(fd, vreshape)
        else:
            raise Exception(f"fd must be a single number or an array-type of length {data.shape[axis]} for this data...")

    # generate time vector same length as data.shape[axis] sampled at fs
    tn = tstart + np.arange(start=0, stop=data.shape[axis], step=1)/fs # [s]
    tn = np.reshape(tn, vreshape)

    # generate sin/cos to demodulate frequency
    I_unfiltered =  2*np.cos(2*np.pi*fd*tn) * data
    Q_unfiltered = -2*np.sin(2*np.pi*fd*tn) * data

    # Low pass filter
    if 2*fc >= fs:
        logger.warn(f"fc({fc}) is too large to filter with the given fs({fs}). Returning unfiltered IQ data")
        return I_unfiltered, Q_unfiltered
    
    ## Generate a window filter
    # find size of window based on the cutoff and sampling frequencies
    n_half = int((data.shape[axis]-1) * fc/fs - 1)
    n_total = int(2*n_half + 1)
    
    # build a frequency domain LPF from the window
    window = sig.get_window(filter, n_total)
    filter = np.zeros_like(tn).flatten()
    filter[0:(n_half+1)] = window[n_half:n_total]
    filter[-(n_half+1):-1] = window[0:n_half]
    shape = np.ones(len(data.shape), dtype=int)
    shape[axis] = -1
    filter = np.reshape(filter, shape)

    # filter data in frequency domain
    I_freq_filtered = np.fft.fft(I_unfiltered, axis=axis) * filter
    Q_freq_filtered = np.fft.fft(Q_unfiltered, axis=axis) * filter

    # take ifft of data
    I_filtered = np.fft.ifft(I_freq_filtered, axis=axis)
    Q_filtered = np.fft.ifft(Q_freq_filtered, axis=axis)

    return np.real(I_filtered), np.real(Q_filtered)

def remodulate(I, Q, fd, fs, tstart=0, axis=0, filter='tukey'):
    """demodulates data, a numpy tensor, along the given 'axis'
    
    Paramaters:
        I: in-phase data
        Q: quadrature data
        fd: demodulation frequency (must be a single number or have the same length as data.shape[axis])
        fs: sampling frequency [Hz] along 'axis'
        tstart: the time associated with the first sample along 'axis'
        axis: the axis along which to demodulate
        filter: the type of low pass filter used
    Returns:
        rf: real component of rf signal
    """
    logger.info("Demodulating  via the _demodulate_numpy method...")

    # check that axis is within bounds of data
    if np.ndim(I) <= axis or np.ndim(Q) <= axis :
        raise Exception(f"Axis is {axis}. Must be < {np.ndims(I)} for this data...")

    # make a reshaping vector for tn and fd (if fd is a vector)
    vreshape = np.ones(len(I.shape), dtype=int)
    vreshape[axis] = I.shape[axis]

    # validate that fd is either a single number or an array of length data.shape[axis]
    if np.ndim(fd)!=0:
        if len(fd) == 1:
            fd = float(fd)
        elif len(fd) == I.shape[axis]:
            fd = np.reshape(fd, vreshape)
        else:
            raise Exception(f"fd must be a single number or an array-type of length {I.shape[axis]} for this data...")

    # generate time vector same length as data.shape[axis] sampled at fs
    tn = tstart + np.arange(start=0, stop=I.shape[axis], step=1)/fs # [s]
    tn = np.reshape(tn, vreshape)

    # convert to rf signal
    rf = I * np.cos(2*np.pi*fd*tn) - Q * np.sin(2*np.pi*fd*tn)
    return rf


