import logging as __logging
logger = __logging.getLogger(__name__)

__XCORR_EPSILON__ = 1E-64

def calc_xcorr_pair(ref, iir:int, search, iis:int, fs=None):
    """Calculate the shift of a reference kernel in a larger signal""" 
    from scipy.signal import correlate
    from dispest.exceptions import XCorrException
    import numpy as np

    if len(ref) > len(search): raise XCorrException("Reference kernel is larger than the search kernel")
    
    ## Actual nxcorr part - use of correlate assumes zero mean
    # Calculate the cross correlation of the ref to signal
    cross = correlate(search, ref, mode='valid')
    # Calculate the autocorrelation for each signal 
    selfa = np.sum(np.abs(ref)**2)
    selfb = np.convolve(np.abs(search)**2, np.ones(len(ref)), mode='valid')

    ## Guard against zero errors
    # Raise an error if reference kernel is zeros
    if (np.abs(selfa) <= __XCORR_EPSILON__): raise XCorrException("Reference kernel is all zeros")
    
    # Supress results if region of search is too zero-like
    _selflow = np.abs(selfb) <= __XCORR_EPSILON__
    cross[_selflow] = 0
    selfb[_selflow] = 1
    rho = cross/np.sqrt(selfa*selfb)

    # find the index of the peak in the correlation function
    irho = np.argmax(np.abs(rho))
    logger.info(f"Peak found at index {irho}")

    # ensure point is within search range - cant poly fit if at edge
    if (irho == 0) and (irho == len(rho)-1):
        raise XCorrException("Max was found at edge of correlation")
    
    # estimate time and value of peak with quadratic fit
    fit = np.polyfit(np.arange(irho-1, irho+2), rho[(irho-1):(irho+2)], deg=2)
    ipeak = -fit[1]/(2*fit[0])
    rho_max = np.polyval(fit, ipeak)

    # convert to shifted index
    di = iir - iis - ipeak

    # shift returned in units of indices
    if fs is None:
        logger.info("No sampling frequency given. Units are in indices")
        return di, rho_max
    
    # shift returned in units of time
    logger.info("Converting shift to time.")
    return di/fs, rho_max

def calc_xcorr_pair_mu(ref, iir:int, search, iis:int, fs=None):
    """Calcculate normalized cross correlation"""
    from dispest.exceptions import XCorrException
    import numpy as np
    if (np.ndim(ref) != 1) or (np.ndim(search) != 1): raise XCorrException('Reference and search vectors must be 1D')

    if len(ref) > len(search)+2: raise XCorrException("Reference kernel is larger than the search kernel")

    Nref = len(ref)
    Ncomp = len(search) - Nref

    # reslice the pair
    it = np.arange(Nref, dtype=int).reshape(1,-1)
    ilag = np.arange(Ncomp, dtype=int).reshape(-1,1)
    indices = it + ilag
    sersel = search[indices]

    # format matrices
    __mu = np.mean(sersel, axis=1).reshape(-1,1)
    __search = sersel - __mu
    __ref = ref.reshape(1,-1) - np.mean(ref)

    # calculate cross correlation terms
    __cross = np.sum(__search * __ref, axis=1)
    __selfr = np.sum(np.abs(__ref)**2, axis=1)
    __selfs = np.sum(np.abs(__search)**2, axis=1)

    __denom = np.sqrt(__selfr * __selfs)

    rho = __cross/__denom
    
    irho = np.argmax(np.abs(rho))
    logger.info(f"Peak found at index {irho}")

    # ensure point is within search range - cant poly fit if at edge
    if (irho == 0) and (irho == len(rho)-1):
        raise XCorrException("Max was found at edge of correlation")
    
    # estimate time and value of peak with quadratic fit
    fit = np.polyfit(np.arange(irho-1, irho+2), rho[(irho-1):(irho+2)], deg=2)
    ipeak = -fit[1]/(2*fit[0])
    rho_max = np.polyval(fit, ipeak)

    # convert to shifted index
    di = iir - iis - ipeak

    # shift returned in units of indices
    if fs is None:
        logger.info("No sampling frequency given. Units are in indices")
        return di, rho_max
    
    # shift returned in units of time
    logger.info("Converting shift to time.")
    return di/fs, rho_max