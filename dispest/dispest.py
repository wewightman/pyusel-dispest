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

def calc_disp_nxcor(RF, taxis: int = 2, fs = None, fd = None, c: float = 1540.0, kusf:int=4, ksize: int = 16, searchsize:int=32, stepsize=4, kssaxis: int = 0, progressive=True, mode='diff', kind=3):
    """Calculte small scale displacement via Kasai algorithm

    Parameters:
    ----
    RF          : signal tensor, real only
    taxis       : axis of tensor over which to calculate phase shift (big-time)
    fs          : sampling frequency in Hz
    fd          : demodulation frequency in Hz
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
    """

    raise Exception("Method has not been implemented")

    # determine the output scale: samples, seconds, or um
    if (fs is None):
        logger.info("Scale is in input samples")
        _scale = 1/kusf
    elif (fs is not None) and (fd is None):
        logger.info("Scale is in time [us]")
        _scale = 1E6/(kusf*fs)
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
    else:
        logger.info('Using fixed reference')
        _t0_slicer[taxis] = slice(1)
    _t0_slicer = tuple(_t0_slicer)
    _rf_0 = RF[_t0_slicer]

    # Isolate t(1) to t(N)
    _t1_slicer = [slice(dim) for dim in RF.shape]
    _t1_slicer[taxis] = slice(1, RF.shape[taxis])
    _t1_slicer = tuple(_t1_slicer)
    _rf_1 = RF[_t1_slicer]

    # calculate the number of XCOR pixels to calculate in a given timeline
    _nxcor = int(RF.shape[kssaxis]//stepsize)

    if _nxcor < 1:
        raise ValueError(f"length of kssaxis is too short -- cannot calculate displacement with step size of {stepsize//kusf}")

    # get the reference start and stop indices
    _ref_start = stepsize*np.arange(_nxcor, dtype=int) - int(ksize//2)
    _ref_start_offset = np.zeros(_ref_start.shape, dtype=int)
    _ref_start_offset[_ref_start < 0] = -_ref_start[_ref_start < 0]
    _ref_start[_ref_start < 0] = 0
    _ref_stop = stepsize*np.arange(_nxcor, dtype=int) - int(ksize//2) + ksize
    _ref_stop_offset = ksize*np.ones(_ref_stop.shape, dtype=int)
    _ref_stop_offset[_ref_stop > RF.shape[kssaxis]] = ksize - (_search_stop[_search_stop  > RF.shape[kssaxis]] - RF.shape[kssaxis])
    _ref_stop[_ref_stop > RF.shape[kssaxis]] = RF.shape[kssaxis]

    # get the search start and stop indices
    _search_start = stepsize*np.arange(_nxcor, dtype=int) - int(searchsize//2)
    _search_start_offset = np.zeros(_search_start.shape, dtype=int)
    _search_start_offset[_search_start < 0] = -_ref_start[_search_start < 0]
    _search_start[_search_start < 0] = 0
    _search_stop = stepsize*np.arange(_nxcor, dtype=int) - int(searchsize//2) + searchsize
    _search_stop_offset = searchsize*np.ones(_search_stop.shape, dtype=int)
    _search_stop_offset[_search_stop > RF.shape[kssaxis]] = searchsize - (_search_stop[_search_stop  > RF.shape[kssaxis]] - RF.shape[kssaxis])
    _search_stop[_search_stop > RF.shape[kssaxis]] = RF.shape[kssaxis]

    # determine the shapes of the intermediate buffers - make kssaxis _nxcor long and tack on the buffer size
    _refs_shape = _rf_0.shape
    _refs_shape[kssaxis] = _nxcor
    _refs_shape = (*_refs_shape, ksize)

    _searches_shape = _rf_1.shape
    _searches_shape[kssaxis] = _nxcor
    _searches_shape = (*_searches_shape, searchsize)

    # fill the intermediate buffers
    _refs = np.zeros(_refs_shape)
    _searches = np.zeros(_searches_shape)

    # build slicing objects
    _refs_slices = [slice(dim) for dim in _refs.shape]
    _searches_slices = [slice(dim) for dim in _searches.shape]

    for _ixc in range(_nxcor):
        # set the last slicing object to match data selection
        _refs_slices[kssaxis] = slice(_ref_start[_ixc], _ref_stop[_ixc])
        _refs_slice = tuple(_refs_slices)


