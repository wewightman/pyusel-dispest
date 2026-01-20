#include "__xcorr__.h"

void calc_nxc_and_std_lagpairs(
    long long       nx,     // number of samples in the signal
    long long       nref0,  // number of starting indices to consider
    long long *     ref0s,  // start index for reference kernels
    long long       lref,   // number of samples in reference kernel
    long long       nsrc,   // number of samples to search +/- ref0
    const float *   refsig, // reference signal
    const float *   srcsig, // search signal
    float *         lag,    // output buffer for lags
    float *         rho,    // output buffer for correlation coefficents
    float *         refstd, // output buffer for STD of reference kernel
    float *         srcstd, // output buffer for STD of search kernel
    float           epsilon // bias to regularize NXC denominator
)
{
    long long iref, ix0, i, isrc0min, isrc0max, irho2max, ilag;
    float rho2max, cross, vara, varb, mua, mub, flref;

    for (iref = 0; iref < nref0; ++iref)
    {
        isrc0min = (ref0s[iref]-nsrc      >= 0) ? -nsrc : -ref0s[iref];
        isrc0max = (ref0s[iref]+nsrc+lref < nx) ?  nsrc : nx-ref0s[iref]-lref-1;

        ix0 = ref0s[iref];

        rho2max = -1.0;
        irho2max = isrc0min;

        // mean loop
        mua = vara = 0.0f;
        for (i = 0; i < lref; ++i) mua += refsig[ix0+i];
        mua /= flref;
        // variance loop
        for (i = 0; i < lref; ++i) vara += (refsig[ix0+i]-mua) * (refsig[ix0+i]-mua);
        vara /= flref;

        // pass through once to find the coordinates of the peak
        for (ilag = isrc0min; ilag < isrc0max+1; ++ilag) 
        {
            mub = varb = cross = 0.0f;

            // calculate search kernel mean
            for (i = 0; i < lref; ++i) mub += srcsig[ix0 + ilag + i];
            mub /= flref;

            // calcualte search kernel variance and search x reference covariance
            for (i = 0; i < lref; ++i) 
            {
                varb  += (srcsig[ix0 + ilag + i] - mub) * (srcsig[ix0 + ilag + i] - mub);
                cross += (srcsig[ix0 + ilag + i] - mub) * (refsig[ix0        + i] - mua);
            }
            varb  /= flref;
            cross /= flref;
            
            // update current max rho^2 value and lag index
            if (rho2max < cross/(sqrt(vara*varb) + epsilon))
            {
                rho2max = cross/(sqrt(vara*varb) + epsilon);
                irho2max = ilag;
            }
        }
    
        // return zeros if the max was at the edge of the search region
        if ((irho2max == isrc0min) || (irho2max == isrc0max+1))
        {
            rho   [iref] = 0.0f;
            lag   [iref] = 0.0f;
            refstd[iref] = 0.0f;
            srcstd[iref] = 0.0f;
        }

        // else intepolate the peak NCC, signal STDs, and lag
        else
        {
            // allocate buffers
            float rhomax[3];

            for (ilag = irho2max - 1; ilag <= irho2max + 1; ++ilag)
            {
                mub = varb = cross = 0.0f;

                // calculate search kernel mean
                for (i = 0; i < lref; ++i) mub += srcsig[ix0 + ilag + i];
                mub /= flref;

                // calcualte search kernel variance and search x reference covariance
                for (i = 0; i < lref; ++i) 
                {
                    varb  += (srcsig[ix0 + ilag + i] - mub) * (srcsig[ix0 + ilag + i] - mub);
                    cross += (srcsig[ix0 + ilag + i] - mub) * (refsig[ix0        + i] - mua);
                }
                varb  /= flref;
                cross /= flref;

                rhomax[ilag - irho2max + 1] = cross / sqrt(epsilon + vara*varb);
            }

            // fit three points to a quadratic to find sub-sample lag (reuse mua and mub variables)
            mua = (rhomax[0] + rhomax[2])/2 - rhomax[1];
            mub = (rhomax[2] - rhomax[0])/2;
            // c = rhomax[1]

            cross = - mub/(2*mua);

            // calculate index+subindex lag, inteprolated NCC, and STDs for each kernel
            lag[iref] = ((float) irho2max) + cross;
            rho[iref] = mua * cross * cross + mub * cross + rhomax[1];
            refstd[iref] = sqrt(vara);
            srcstd[iref] = sqrt(varb);
        }

        if (abs(rho[iref]) > 1) rho[iref] = (rho[iref] > 0) ? 1.0f : -1.0f;
    }    
}

/**
 * calc_nxc_and_std_lagpairs: calculate the sliding window normalized cross correlation
 */
void calc_nxc_lagpairs(
    long long       nx,     // number of samples in the signal
    long long       nref0,  // number of starting indices to consider
    long long *     ref0s,  // start index for reference kernels
    long long       lref,   // number of samples in reference kernel
    long long       nsrc,   // number of samples to search +/- ref0
    const float *   refsig, // reference signal
    const float *   srcsig, // search signal
    float *         lag,    // output buffer for lags
    float *         rho,    // output buffer for correlation coefficents
    float           epsilon // bias to regularize NXC denominator
)
{
    long long iref, ix0, i, isrc0min, isrc0max, irho2max, ilag;
    float rho2max, cross, vara, varb, mua, mub, flref;

    for (iref = 0; iref < nref0; ++iref)
    {
        isrc0min = (ref0s[iref]-nsrc      >= 0) ? -nsrc : -ref0s[iref];
        isrc0max = (ref0s[iref]+nsrc+lref < nx) ?  nsrc : nx-ref0s[iref]-lref-1;

        ix0 = ref0s[iref];

        rho2max = -1.0;
        irho2max = isrc0min;

        // mean loop
        mua = vara = 0.0f;
        for (i = 0; i < lref; ++i) mua += refsig[ix0+i];
        mua /= flref;
        // variance loop
        for (i = 0; i < lref; ++i) vara += (refsig[ix0+i]-mua) * (refsig[ix0+i]-mua);
        vara /= flref;

        // pass through once to find the coordinates of the peak
        for (ilag = isrc0min; ilag < isrc0max+1; ++ilag) 
        {
            mub = varb = cross = 0.0f;

            // calculate search kernel mean
            for (i = 0; i < lref; ++i) mub += srcsig[ix0 + ilag + i];
            mub /= flref;

            // calcualte search kernel variance and search x reference covariance
            for (i = 0; i < lref; ++i) 
            {
                varb  += (srcsig[ix0 + ilag + i] - mub) * (srcsig[ix0 + ilag + i] - mub);
                cross += (srcsig[ix0 + ilag + i] - mub) * (refsig[ix0        + i] - mua);
            }
            varb  /= flref;
            cross /= flref;
            
            // update current max rho^2 value and lag index
            if (rho2max < cross/(sqrt(vara*varb) + epsilon))
            {
                rho2max = cross/(sqrt(vara*varb) + epsilon);
                irho2max = ilag;
            }
        }
    
        // return zeros if the max was at the edge of the search region
        if ((irho2max == isrc0min) || (irho2max == isrc0max+1))
        {
            rho   [iref] = 0.0f;
            lag   [iref] = 0.0f;
        }

        // else intepolate the peak NCC, signal STDs, and lag
        else
        {
            // allocate buffers
            float rhomax[3];

            for (ilag = irho2max - 1; ilag <= irho2max + 1; ++ilag)
            {
                mub = varb = cross = 0.0f;

                // calculate search kernel mean
                for (i = 0; i < lref; ++i) mub += srcsig[ix0 + ilag + i];
                mub /= flref;

                // calcualte search kernel variance and search x reference covariance
                for (i = 0; i < lref; ++i) 
                {
                    varb  += (srcsig[ix0 + ilag + i] - mub) * (srcsig[ix0 + ilag + i] - mub);
                    cross += (srcsig[ix0 + ilag + i] - mub) * (refsig[ix0        + i] - mua);
                }
                varb  /= flref;
                cross /= flref;

                rhomax[ilag - irho2max + 1] = cross / sqrt(epsilon + vara*varb);
            }

            // fit three points to a quadratic to find sub-sample lag (reuse mua and mub variables)
            mua = (rhomax[0] + rhomax[2])/2 - rhomax[1];
            mub = (rhomax[2] - rhomax[0])/2;
            // c = rhomax[1]

            cross = - mub/(2*mua);

            // calculate index+subindex lag, inteprolated NCC, and STDs for each kernel
            lag[iref] = ((float) irho2max) + cross;
            rho[iref] = mua * cross * cross + mub * cross + rhomax[1];
        }

        if (abs(rho[iref]) > 1) rho[iref] = (rho[iref] > 0) ? 1.0f : -1.0f;
    }
}