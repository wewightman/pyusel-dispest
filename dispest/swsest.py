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
    imax = np.where(sums == np.max(sums))
    if len(imax[0]) != 2:
        st1, st2 = imax
    else:
        st1, st2 = imax[0]

    c = float((latmax-latmin)/(trange[st2]-trange[st1]))

    if speedonly:
        return c
    else:
        return c, sums, trange