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
    import numpy as np

    trange = np.linspace(tmin, tmax, N)
    lat = np.array(lat)
    latmask = (lat >= latmin) & (lat <= latmin)
    # f_set = [i1d()]

    # f_set = RGI((lat, t), spctm, method='cubic', bounds_error=False, fill_value=0)
    # sums = np.zeros((N,N))

    # for it1 in range(N):
    #     for it2 in range(it1+1, N):
    #         teval = np.linspace(trange[it1], trange[it2], N)
    #         eval = np.array([teval, lateval]).T
    #         sums[it1, it2] = np.sum(f(eval))
    
    # imax = np.where(sums == np.max(sums))
    # st1, st2 = imax[0]

    # c = (latmax-latmin)/(teval[st2]-teval[st1])

    # if speedonly:
    #     return c
    # else:
    #     return c, sums, teval
    
