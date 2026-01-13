import logging as __logging
logger = __logging.getLogger(__name__)

__XCORR_EPSILON__ = 1E-64

def get_xcorr_inds(Ns:int, lenref:int, refstep:int, searchpm:int, istart:int=0, istop:int|None=None):
    """Generate indices used to select reference and search kernels from two signals
    
    # Parameters
    `Ns`: number of samples in each signal
    `lenref`: length of the reference kernel in samples
    `refstep`: number of samples between the beginning of each reference kernel
    `searchpm`: how many samples to search from the refernce kernel in either direction
    `istart`: center point of the first reference kernel
    `istop`: the center point of the last reference kernel to consider

    # Returns
    `selref`: indices of a 1D reference signal that correspond to all needed reference kernels
    `selser`: indices of a 1D search signal that correspond to all needed search kernels
    `outbnd`: boolean matrix indicating which indices in selser are not within the bounds of the search signal
    `seliref`: shaped matrix used to slice the rows of the correlation matrix
    `imid`: the indical midpoint of each reference kernel
    """

    import numpy as np

    # make indices list that is the length of the kernel
    kern = np.arange(lenref, dtype=int)

    # make indices list that finds the first index of each reference kernel
    istart = max(int(istart-lenref//2), 0)
    if istop is not None: istop = min(Ns-lenref, istop)
    else: istop = Ns-lenref
    iref_start = np.arange(istart, istop, refstep, dtype=int)

    # Calculate int midpoint of each reference kernel in indices
    imid = iref_start + lenref/2

    # make indices list for how far to search from the reference kernel index
    iserpm = np.arange(-searchpm, searchpm+1, dtype=int)

    # broadcast ref_start and kern to get a N by 1 by k shaped list of indices
    selref = iref_start[:,None, None] + kern[None,None,:]

    # broadcast ref_start, kern, and isearpm to get a N by 2*searchpm+1 by k shaped list of indices
    selser = selref + iserpm[None,:,None]

    # find and build a mask for all search kernel indices that are out of bound 
    outbnd = (selser < 0) | (selser >= Ns)
    selser[outbnd] = 0

    # make a list to select the proper rows and three collumns from the N by 2*searchpm+1 correlation matrix
    seliref = np.arange(selref.shape[0], dtype=int)[None,:] * np.ones((3,1), dtype=int)

    return selref, selser, outbnd, seliref, imid

def nxcorr_by_inds_mu(sigref, sigsearch, selref, selser, outbnd, seliref, get_power=False, bias:float=0):
    """Calculate the normalized cross correlation coefficients between two signals with the specified kernels"""
    import numpy as np

    # extract all kernels from reference and search
    REF = sigref[selref]
    SER = sigsearch[selser]

    # Mask out boundary conflicted values
    SER[outbnd] = np.nan

    # calculate searchpm from selser
    searchpm = int((SER.shape[1]-1)//2)

    # subtract the mean and get the power of the reference kernels
    REF -= np.mean(REF, axis=2).reshape(REF.shape[0], REF.shape[1], 1)
    REF_STD = np.std(REF, axis=2)

    # subtract the mean and get the power of the search kernels
    SER -= np.mean(SER, axis=2).reshape(SER.shape[0], SER.shape[1], 1)
    SER_STD = np.std(SER, axis=2)

    # calculate the cross correlation between REF and SER
    CROSS = np.mean(REF * SER, axis=2)

    # normalize to the combined signal power
    RHOS = CROSS/(bias + SER_STD*REF_STD)

    # get the peak correlation coefficients
    imax = np.nanargmax(RHOS, axis=1)
    invbnd = (imax < 1) | (imax >= 2*searchpm)
    imax[invbnd] = searchpm
    sellag = imax[None,:] + np.array([[-1, 0, 1]], dtype=int).T
    peaks = RHOS[seliref, sellag]

    # fit a quadratic to the peak correlation coefficient and its neighbors
    a = (peaks[0] + peaks[2])/2 - peaks[1]
    b = (peaks[2] - peaks[0])/2
    c = peaks[1]

    # Estimate the shift in peak location predicted by the quadratic fit
    dmax = -b/(2*a)

    # find the correlation coefficient at the peak 'index'
    rhomax = a*dmax*dmax + b*dmax + c

    # Combine the predicted kernel index and predicted shift to get the true shift relative to the reference kernel
    ilag = imax-searchpm

    # mask out-of-bounds estimates
    dmax[invbnd] = 0
    rhomax[invbnd] = 0
    ilag[invbnd] = 0

    if not get_power: return ilag, dmax, rhomax
    
    # isolate signal strength at ilag
    ref_pow = REF_STD.squeeze()
    ser_pow = SER_STD[np.arange(SER.shape[0],dtype=int), imax].squeeze()

    # mask out-of-bounds estiamtes
    ser_pow[invbnd] = 0
    ref_pow[invbnd] = 0


    return ilag, dmax, rhomax, ref_pow, ser_pow

def nxcorr_gpu(
        sigref, sigser, 
        lenref:int, refstep:int, searchpm:int,
        istart:int=0, istop:int|None=None,
        get_power:bool=False, bias:float=0,
        retasnp:bool=True, ntpb:int=256
    ):
    """Calculate the sliding window normalized cross correlation using GPU
    
    # Parameters
    - `sigref,sigser`: the reference and search signals with length `Ns` last axis
    - `lenref`: signal kernel length in samples
    - `refstep`: steps between reference kernels in samples
    - `searchpm`: the plus/minus search region in samples
    - `istart`: the optional starting index of the first kernel. Default zero
    - `istop`: the optional starting index of the last kernel, Default `Ns-lenref`
    - `get_power`: whether to return the signal powers or not
    - `bias`: the bias to use in the denominator of NXCC calculation to prevent divide by zero. Default zero
    - `retasnp`: boolean indicatign whether to return as numpy array. Default True
    - `ntpb`: number of threads per GPU block

    # Returns
    - `lag`: Lag estimate in units of samples
    - `rho`: normalized correlation coefficient
    - `imid`: the indical midpoint of each kernel
    - `ref_std`: the standard deviation of the reference signal within the kernel
    - `ser_std`: the standard deviation of the search signal within the lag-matched kernel
    """

    import os
    import cupy as cp
    import numpy as np

    # check that inputs are valid
    if (sigref.shape != sigser.shape): raise Exception("sigref and sigser must be the same shape")
    

    # calculate nvec and ns for potnetial upsampling nxc
    Ns = np.int64(sigref.shape[-1])
    Nvec = np.int64(np.prod(sigref.shape[:-1]))

    # find the first index of each reference kernel
    istart = max(int(istart-lenref//2), 0)
    if istop is not None: istop = min(Ns-lenref, istop)
    else: istop = Ns-lenref
    iref_start = np.arange(istart, istop, refstep, dtype=int)

    nref0 = np.int64(len(iref_start))

    # Calculate midpoint of each reference kernel in indices
    imid = iref_start + lenref/2

    # load the cuda nxc module
    __base_eng_path__ = os.path.join(os.path.dirname(__file__), "__xcorr__.cu")
    with open(__base_eng_path__, mode='r') as fp: raw_module_str = fp.read()
    __base_eng__ = cp.RawModule(code=raw_module_str)

    calc_nxc_lagpairs    = __base_eng__.get_function("calc_nxc_lagpairs")

    lag = cp.zeros((*sigref.shape[:-1], len(iref_start)), dtype=np.float32)
    rho = cp.zeros(lag.shape, dtype=np.float32)
    ref_std = cp.zeros(lag.shape, dtype=np.float32)
    ser_std = cp.zeros(lag.shape, dtype=np.float32)

    params = (
        nref0,
        cp.array(iref_start, dtype=np.int64),
        np.int64(lenref),
        np.int64(searchpm),
        Ns,
        Nvec,
        cp.ascontiguousarray(cp.array(sigref), dtype=np.float32),
        cp.ascontiguousarray(cp.array(sigser), dtype=np.float32),
        rho,
        lag,
        ref_std,
        ser_std,
        np.float32(bias)
    )

    nblocks = int(np.ceil(Nvec*nref0/ntpb))

    calc_nxc_lagpairs((nblocks,1,1), (ntpb,1,1), params)

    if   get_power and retasnp:
        return cp.asnumpy(lag), cp.asnumpy(rho), imid, cp.asnumpy(ref_std), cp.asnumpy(ser_std)
    elif get_power:
        return lag, rho, imid, ref_std, ser_std
    elif retasnp:
        return cp.asnumpy(lag), cp.asnumpy(rho), imid
    else:
        return lag, rho, imid

