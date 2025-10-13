# dispest
This repository contains python-based particle displacement estimators, some demodulation tools, and some rudimentary SWS estimators.

## `calc_kasai`
This function calculates particle motion via the Kasai algorithm

## `swsradon`
This function calculates the shear wave speed on a lateral-temporal space-time plot.

## `demod_tensor` and `remod_tensor`
These functions use an FFT-based approach to demodulate and remodulate an N-dimensional array of data along a given axis.
Super fast!

## `get_xcorr_inds` and `nxcorr_by_inds_mu`
These two functions are needed for (relatively) fast computation of mean-subtracted, sliding-window normalized cross correlation with NumPy.
`get_xcorr_inds` precomputes the indices to slice a 1D signal for the broadcastable calculation of sliding window normalized cross correlation that occurs in `nxcorr_by_inds_mu`.