import logging as __logging
logger = __logging.getLogger(__name__)

def swsradon(spctm, lat, t, latmin, latmax, tmin, tmax, N:int=512, speedonly:bool=True):
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
    from scipy.interpolate import RectBivariateSpline as RBS
    import numpy as np

    t = np.array(t).flatten()
    trange = np.linspace(tmin, tmax, N)

    # Find the number of lateral positions contained within given range
    lat = np.array(lat).flatten()
    latmask = (lat >= latmin) & (lat <= latmax)
    nlat = np.sum(latmask)
    if nlat == 0:
        raise ValueError(f"Latmask range of {latmin} to {latmax} does not contain any points")
    latmin = np.min(lat[latmask])
    latmax = np.max(lat[latmask])
    lateval = np.linspace(latmin, latmax, nlat)

    f = RBS(lat, t, spctm)

    sums = np.zeros((N,N))

    for it1 in range(N):
        for it2 in range(it1+1, N):
            teval = np.linspace(trange[it1], trange[it2], nlat)
            sums[it1, it2] = np.sum(f(lateval, teval, grid=False))
    
    imax = np.where(sums == np.max(sums))
    if len(imax[0]) != 2:
        st1, st2 = imax
    else:
        st1, st2 = imax[0]

    c = (latmax-latmin)/(trange[st2]-trange[st1])

    if speedonly:
        return c
    else:
        return c, sums, trange
    
