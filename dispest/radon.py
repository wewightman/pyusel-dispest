import logging as __logging
logger = __logging.getLogger(__name__)

def swsradon(spctm, lat, t, latmin, latmax, tmin, tmax, N:int=512, speedonly:bool=True, interpkind='cubic'):
    """Calculate the shear wave speed using radon sum algorithm
    
    Parameters:
    ----
    `spctm`: space time plot with displacement/velocity data [MxN]
    `lat`: length M lateral location vector of every row of `spctm` in mm
    `t`: length N time vector in ms
    `latmin`: minimum value of lateral extent to evaluate
    `latmax`: maximum value of lateral extent to evaluate
    `tmin`: minimum value of temporal extent to evaluate
    `tmax`: maximum value of temporal extent to evaluate
    `N`: Number of time points to evaluate along lateral and temporal extents
    `speedonly`: determines whether to return just speed or (speed, radonsum, t)

    Returns:
    ----
    `speed`: the estimated speed in m/s
    or 
    `(speed, radonsum, t)`: estimated speed in m/s, energy integration chart, and time vector along the edges
    """
    from scipy.interpolate import interp1d as i1d
    import numpy as np

    t = np.array(t).flatten()
    trange = np.linspace(tmin, tmax, N)

    # Find the number of lateral positions contained within given range
    lat = np.array(lat).flatten()
    latmask = (lat >= latmin) & (lat <= latmax)
    lat = lat[latmask]
    nlat = np.sum(latmask)
    if nlat == 0:
        raise ValueError(f"Latmask range of {latmin} to {latmax} does not contain any points")
    latmin = np.min(lat)
    latmax = np.max(lat)
    spctm = spctm[latmask,:]

    t1 = np.tile(trange.reshape((-1,1)), (1, N))
    t2 = np.tile(trange.reshape((1,-1)), (N, 1))

    sums = np.zeros((N,N))
    for ilat in range(nlat):
        tsel = (lat[ilat] - latmin)*(t2-t1)/(latmax-latmin) + t1
        f = i1d(t, spctm[ilat,:], kind=interpkind, bounds_error=False, fill_value=0)
        sums += f(tsel)

    sums[t1 >= t2] = 0
    st1, st2 = np.where(sums == np.max(sums))
    st1 = st1[0]
    st2 = st2[0]

    validst1 = (st1 >= 1) | (st1 < N-1)
    validst2 = (st2 >= 1) | (st2 < N-1)
    if not (validst1 and validst2):
        raise Exception(f"Both temporal coordinates must be between 1 and {N-1}, but found {st1} and {st2}")
    
    # estimate the sub-index peak
    Ts = trange[1]-trange[0]
    peakst1 = sums[(st1-1):(st1+2), st2]
    dt1 = Ts * quadfitreg(peakst1)
    peakst2 = sums[st1, (st2-1):(st2+2)]
    dt2 = Ts * quadfitreg(peakst2)
    
    dt12 = np.mean(trange[st2] + dt2 - trange[st1] - dt1)
    c = float((latmax-latmin)/dt12)

    if speedonly:
        return c
    else:
        return c, sums, trange

def quadfitreg(peaks):
    """Quadratic fit assuming reular sampling"""

    a = (peaks[0] + peaks[2])/2 - peaks[1]
    b = (peaks[2] - peaks[0])/2
    c = peaks[1]

    return -b/(2*a)