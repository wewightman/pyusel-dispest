"""Displacment estimators"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calc_kasai(I, Q, taxis: int = 2, fd = None, c: float = 1540.0, ksize: int = 1, kaxis: int = 0, progressive=True, mode='cumsum'):
    """Calculte small scale displacement via Kasai algorithm

    Parameters:
    ----
    `I`: in phase component of signal
    `Q`: quadrature component of the signal
    `taxis`: axis of tensor over which to calculate phase shift (big-time)
    `c`: speed of sound in m/s
    `ksize`: size of the averaging kernel in pixels
    `kaxis`: axis over wich to apply kernel (fast time)
    `fd`: demodulation frequency in Hz
    `mode`: 'cumsum' or 'differential'

    Returns:
    ----
    `disp`: displacement information
    """

    # Scales by wavelength if system information is provided
    if fd is None:
        _scale = 1
        logger.info('Displacements returned in radians.')
    else:
        _scale = (c / 2) * 1e6 / (2 * np.pi * fd)
        logger.info('Scaling returned displacements to microns.')

    # Isolate t(0) to t(N-1)
    _t0_slicer = [slice(I.shape[ind]) for ind in range(len(I.shape))]
    if progressive:
        logger.info('Using progressive reference')
        _t0_slicer[taxis] = slice(I.shape[taxis]-1)
        shape = [*(I.shape)]
        shape[taxis] = I.shape[taxis]-1
    else:
        logger.info('Using fixed reference')
        _t0_slicer[taxis] = slice(1)
        shape = [*(I.shape)]
        shape[taxis] = 1
    _t0_slicer = tuple(_t0_slicer)
    _i_0 = I[_t0_slicer]
    _q_0 = Q[_t0_slicer]

    logger.debug("_i_0 shape" + str(_i_0.shape) + "I" + str(I.shape))

    # Isolate t(1) to t(N)
    _t1_slicer = [slice(I.shape[ind]) for ind in range(len(I.shape))]
    _t1_slicer[taxis] = slice(1, I.shape[taxis])
    _t1_slicer = tuple(_t1_slicer)
    _i_1 = I[_t1_slicer]
    _q_1 = Q[_t1_slicer]

    # Calculate phase shift info (trig identity for dividing complex numbers)
    _num = _i_1 * _q_0 - _q_1 * _i_0
    _den = _i_0 * _i_1 + _q_0 * _q_1

    # calculate moving of phase shift divisions
    _kshape = np.ones(len(I.shape), dtype=int)
    _kshape[kaxis] = ksize
    _kernel = np.ones(_kshape)/ksize
    if ksize > 1:
        logger.info("Smoothing numerator and denomenator")
        # smooth displacement
        from scipy.signal import fftconvolve
        _num = fftconvolve(_num, _kernel, mode='full', axes=kaxis)
        _den = fftconvolve(_den, _kernel, mode='full', axes=kaxis)

        # fix convolution based offset
        _slices = []
        for dim in I.shape:
            _slices.append(slice(dim))
        
        # find starting and end points along kaxis
        _kstart = int(ksize//2)
        _kend = _kstart + I.shape[kaxis]
        _slices[kaxis] = slice(_kstart, _kend)
        _slices = tuple(_slices)

        # truncate numerator and denomenator
        _num = _num[_slices]
        _den = _den[_slices]
        
    
    # Convert scale phase if demodulation frequency is given
    disp = _scale * np.arctan2(_num, _den)

    if progressive and (mode == 'cumsum'):
        logger.info("Integrating progessive reference")
        disp = np.cumsum(disp, axis=taxis)

    return disp
"""
def calc_disp_nxcor(RF, taxis: int = 2, fs = None, c: float = 1540.0, kusf:int=4, ksize: int = 16, searchsize:int=32, stepsize=4, kssaxis: int = 0, progressive=True, mode='diff', kind=3):
    """"""Calculte small scale displacement via Kasai algorithm

    Parameters:
    ----
    RF          : signal tensor, real only
    taxis       : axis of tensor over which to calculate phase shift (big-time)
    fs          : sampling frequency in Hz
    c           : speed of sound in m/s
    kusf:       : upsample factor along kaxis
    ksize       : size of the correlation kernel in samples, input frequency fs
    searchsize  : size of search kernel in samples, input frequency fs
    stepsize    : size of step kernel in samples, input frequency fs
    kssaxis     : fast time axis over which to use kernel, stepsize, and search size
    progressive : True if progressive referencing is used, false if fixed
    mode        : 'cumsum' or 'differential', only applies if progressive = true
    kind        : polynomial order of interpolation

    Returns:
    ----
    disp: displacement information
    """"""
    # determine the output scale: samples, seconds, or um
    if (fs is None):
        logger.info("Scale is in input samples")
        _scale = 1/kusf
    else:
        logger.info("Scale is in time [um]")
        _scale = 1E6*c/(2*kusf*fs)

    # determine if data should be upsampled or not
    if kusf != 1:
        from scipy.interpolate import interp1d
        # build interpolator function
        _ikaxis = np.arange(RF.shape[kssaxis])
        _ikaxis_usf = np.arange(0, RF.shape[kssaxis], 1/kusf)
        _f_RFusf = interp1d(_ikaxis, RF, kind=kind, bounds_error=False, axis=kssaxis, fill_value=0)
        RF = _f_RFusf(_ikaxis_usf)
        del _f_RFusf

        ksize = int(kusf*ksize)
        searchsize = int(kusf*searchsize)
        stepsize = int(kusf*stepsize)

    # slice data for progressive or fixed referencing
    _t0_slicer = [slice(RF.shape[ind]) for ind in range(len(RF.shape))]
    if progressive:
        logger.info('Using progressive reference')
        _t0_slicer[taxis] = slice(RF.shape[taxis]-1)
        Nt0 = RF.shape[taxis]-1
    else:
        logger.info('Using fixed reference')
        _t0_slicer[taxis] = slice(1)
        Nt0 = 1
    Nsamp = RF.shape[kssaxis]
    _t0_slicer = tuple(_t0_slicer)
    _rf_0 = RF[_t0_slicer]
    _dims_orig = set(range(len(_rf_0.shape)))
    _dims_trans = (*(_dims_orig-{taxis, kssaxis}), taxis, kssaxis)
    _rf_0 = _rf_0.transpose(_dims_trans).flatten(order='c').reshape((-1, Nt0, Nsamp), order='c')

    # Isolate t(1) to t(N)
    _t1_slicer = [slice(dim) for dim in RF.shape]
    _t1_slicer[taxis] = slice(1, RF.shape[taxis])
    Nt1 = RF.shape[taxis]-1
    _t1_slicer = tuple(_t1_slicer)
    _rf_1 = RF[_t1_slicer]
    _rf_1 = _rf_1.transpose(_dims_trans).flatten(order='c').reshape((-1, Nt1, Nsamp), order='c')

    # get reference and search kernels for each point
    for iax in range(int(Nsamp/stepsize)):
        imin = iax*stepsize - ksize//2
        if imin < 0:
            ioffmin = -imin
            imin = 0
        else: ioffmin = 0

        imax = iax*stepsize + ksize - ksize//2
        if imax >= Nsamp:
            ioffmax = ksize - imax + Nsamp
            imax = Nsamp
        else: ioffmax = ksize

        islicer = (slice(None), slice(None), slice(imin, imax))
        ioffslicer = (slice(None), slice(None), slice(ioffmin, ioffmax))

        refbuffer = np.zeros((_rf_0.shape[0], _rf_0.shape[1], ksize))
        refbuffer[ioffslicer] = _rf_0[islicer]

        imin = iax*stepsize - searchsize//2
        if imin < 0:
            ioffmin = -imin
            imin = 0
        else: ioffmin = 0

        imax = iax*stepsize + searchsize - searchsize//2
        if imax >= Nsamp:
            ioffmax = searchsize - imax + Nsamp
            imax = Nsamp
        else: ioffmax = searchsize

        islicer = (slice(None), slice(None), slice(imin, imax))
        ioffslicer = (slice(None), slice(None), slice(ioffmin, ioffmax))

        searchbuffer = np.zeros((_rf_1.shape[0], _rf_1.shape[1], searchsize))
        searchbuffer[ioffslicer] = _rf_1[islicer]

        for ift in range(ksize+searchsize-1):
            imin = 
"""