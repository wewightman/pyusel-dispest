#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef __pycbf_cpu_xcorr__
#define __pycbf_cpu_xcorr__

/**
 * calc_nxc_and_std_lagpairs: calculate the sliding window normalized cross correlation
 */
extern void calc_nxc_and_std_lagpairs(
    long long           nx,     // number of samples in the signal
    long long           nref0,  // number of starting indices to consider
    const long long *   ref0s,  // start index for reference kernels
    long long           lref,   // number of samples in reference kernel
    long long           nsrc,   // number of samples to search +/- ref0
    const float *       refsig, // reference signal
    const float *       srcsig, // search signal
    float *             lag,    // output buffer for lags
    float *             rho,    // output buffer for correlation coefficents
    float *             refstd, // output buffer for STD of reference kernel
    float *             srcstd, // output buffer for STD of search kernel
    float               epsilon // bias to regularize NXC denominator
);

/**
 * calc_nxc_and_std_lagpairs: calculate the sliding window normalized cross correlation
 */
extern void calc_nxc_lagpairs(
    long long           nx,     // number of samples in the signal
    long long           nref0,  // number of starting indices to consider
    const long long *   ref0s,  // start index for reference kernels
    long long           lref,   // number of samples in reference kernel
    long long           nsrc,   // number of samples to search +/- ref0
    const float *       refsig, // reference signal
    const float *       srcsig, // search signal
    float *             lag,    // output buffer for lags
    float *             rho,    // output buffer for correlation coefficents
    float               epsilon // bias to regularize NXC denominator
);

#endif