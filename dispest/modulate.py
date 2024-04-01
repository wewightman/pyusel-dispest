def demod_tensor(rf, fs:float, fmin:float, fmax:float, taxis:int, alpha:float=0.15):
    """Demodulate a tensor of RF data along the specified fast time axis

    Note: runs fastes if fast time axis is last axis in c order array or first axis in f order array
    
    # Parameters:
    `rf`: an ND matrix that has at least `taxis` axes. Data to be demodulated
    `fs`: the sampling frequency along the fast time axis
    `fmin`: the minimum frequency that must be included. Must be in [0, fs/2)
    `fmax`: the maximum frequency that the must be included. Must be (0, fs/2) and greater than `fmin`
    `taxis`: the axis to demodulate along
    `alpha`: the alpha parameter of the tukey window used to zero out the edges of the demodulation pass band. Default 0.15

    # Returns:
    `demod`: an ND complex matrix of demodulated RF data
    `ifmin`: the index of the first frequency included in the window
    `ifmax`: the index of the last frequency included in the window
    `fseff`: the new effective sampling frequency
    """
    from numpy import ceil, ndim
    from numpy.fft import fft, ifft
    from scipy.signal.windows import tukey

    Ndim = int(ndim(rf))

    if Ndim <= taxis:
        raise ValueError(f"rf must have at most (taxis-1)={taxis-1} axes, but has {Ndim}")
    
    Nsamp = rf.shape[taxis]

    ifmin = max(int(fmin*Nsamp/fs), 0)
    ifmax = min(int(ceil(fmax*Nsamp/fs)), int(Nsamp//2))
    
    if ifmax <= ifmin:
        raise ValueError(f"ifmax must be greater than ifmin but were {ifmin} and {ifmax} respectively")

    # Make slicer to get windowed frequency content
    slices = [slice(None)] * Ndim
    slices[taxis] = slice(ifmin, ifmax)

    # Make broadcastable tukey window 
    wshape = [1] * Ndim
    wshape[taxis] = -1
    window = tukey(ifmax-ifmin, alpha=alpha).reshape(*wshape)

    # calculate downsample factor
    dsf = Nsamp / (ifmax-ifmin)

    # take fft of data tensor and select desired data
    RF = fft(rf, axis=taxis)
    RF = window * RF[*slices]
    demod = 2 * ifft(RF, axis=taxis)

    # Calculate the new effective sampling frequency
    fseff = fs / dsf

    return demod, ifmin, ifmax, fseff

def remod_tensor(iq, fs_in:float, fs_out:float, ifmin:int, ifmax:int, taxis:int, return_complex:bool=False):
    """remodulate a tensor of IQ data along the specified fast time axis

    Note: runs fastest if fast time axis is last axis in c order array or first axis in f order array
    
    # Parameters:
    `iq`: an ND matrix that has at least `taxis` axes. Data to be remodulated
    `fs_in`: the input sampling frequency along the fast time axis
    `fs_out`: the desired output sampling frequency along the fast time axis
    `ifmin`: the index of the minimum frequency
    `ifmax`: the index of the maximum frequency
    `taxis`: the axis to demodulate along
    `return_complex`: flag indicating if the full complex signal is returned or just real component

    # Returns:
    `sigout`: the returned signal
    """
    from numpy import ceil, ndim
    from numpy.fft import fft, ifft

    Ndim = int(ndim(iq))

    if Ndim <= taxis:
        raise ValueError(f"rf must have at most (taxis-1)={taxis-1} axes, but has {Ndim}")
    
    Nsamp = iq.shape[taxis]

    dfdn = fs_in/(Nsamp)
    Nfreq = int(np.ceil(fs_out / dfdn))
    fs_out = Nfreq * dfdn

    # calculate the shape of the signal tensor at the new sampling frequency
    sout = [*iq.shape]
    sout[taxis] = Nfreq
    SIGOUT = np.zeros(sout, dtype=complex)

    # Paste the frequency content of the IQ signal into the upsampled array
    slices = [slice(None)] * Ndim
    slices[taxis] = slice(ifmin, ifmax)
    SIGOUT[*slices] = fft(iq, axis=taxis)

    # Calculate the time-space domain representation of the signal
    sigout = (Nfreq/Nsamp) * ifft(SIGOUT, axis=taxis)

    if return_complex: return sigout, fs_out
    return np.real(sigout), fs_out