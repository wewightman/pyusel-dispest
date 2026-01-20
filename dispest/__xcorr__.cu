extern "C" {
    /**
     * Cuda kernels for fast normalized cross correlation
     * The first aplication for these kernels is for sliding window TOF estimation for TR SWV
     * I may also add a particle estimation kernel if it seems to make sense
     */

    // eipsilon value to add to denominator of NXCC to prevent divide by zero errors
    
    __global__ 
    void calc_nxc_and_std_lagpairs(
        long long nref0,    // number of reference kernels to use
        long long * ref0s,  // starting indices of ref. kernels (len nref)
        long long lref,     // num. of samples in each kernel
        long long nsrc,     // num. of samples to search +/- iref
        long long nx,       // num. of samples in each signal
        long long nvec,     // num. of signals to compare
        const float * fref, // ref. signal to compare to (len nvec*nx)
        const float * fsrc, // search signal compared against (len nvec*nx)
        float * rho,        // output norm. xcorr coefs (len nvec*nref)
        float * lag,        // output lag estimates (len nvec*nref)
        float * sigref,     // ref. signal STD in peak kernel (len nvec*nref)
        float * sigsrc,     // search signal STD in peak kernel (len nvec*nref)
        float epsilon       // a non-negative value to bias denominator of NXCC calculation
    ) {
        long long tpb, tid, ivec, iref, ix0, iout;
        float rho2max, cross, vara, varb, mua, mub, flref;
        long long i, isrc0min, isrc0max, irho2max, ilag;
        
        // get cuda step sizes
        tpb = blockDim.x * blockDim.y * blockDim.z; // threads per block

        // Unique thread ID
        tid  = threadIdx.x + threadIdx.y * blockDim.x;
        tid += threadIdx.z * blockDim.x * blockDim.y;
        tid += tpb * (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);

        if (tid >= nvec * nref0) return;

        // get which signal pair and which reference kernel to assess in this function call
        ivec = tid / nref0; // which signal pair is being investigated
        iref = tid % nref0; // which reference kernel is being used

        ix0  = ivec*nx   + ref0s[iref]; // index of first sample of reference kernel
        iout = ivec*nref0 + iref;

        // determine the valid search bounds within this signal
        isrc0min = (ref0s[iref]-nsrc >= 0 ) ? -nsrc : -ref0s[iref];
        isrc0max = (ref0s[iref]+nsrc+lref < nx) ? nsrc : nx - ref0s[iref] - lref-1;

        rho2max = -1.0;
        irho2max = isrc0min;

        // calcualte the mean and STD of the reference kernel (once)
        flref = (float) lref; // float lref
        // mean loop
        mua = vara = 0.0f;
        for (i = 0; i < lref; ++i) mua += fref[ix0+i];
        mua /= flref;
        // variance loop
        for (i = 0; i < lref; ++i) vara += (fref[ix0+i]-mua) * (fref[ix0+i]-mua);
        vara /= flref;

        // pass through once to find the coordinates of the peak
        for (ilag = isrc0min; ilag < isrc0max+1; ++ilag) 
        {
            mub = varb = cross = 0.0f;

            // calculate search kernel mean
            for (i = 0; i < lref; ++i) mub += fsrc[ix0 + ilag + i];
            mub /= flref;

            // calcualte search kernel variance and search x reference covariance
            for (i = 0; i < lref; ++i) 
            {
                varb  += (fsrc[ix0 + ilag + i] - mub) * (fsrc[ix0 + ilag + i] - mub);
                cross += (fsrc[ix0 + ilag + i] - mub) * (fref[ix0        + i] - mua);
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
            rho   [iout] = 0.0f;
            lag   [iout] = 0.0f;
            sigref[iout] = 0.0f;
            sigsrc[iout] = 0.0f;
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
                for (i = 0; i < lref; ++i) mub += fsrc[ix0 + ilag + i];
                mub /= flref;

                // calcualte search kernel variance and search x reference covariance
                for (i = 0; i < lref; ++i) 
                {
                    varb  += (fsrc[ix0 + ilag + i] - mub) * (fsrc[ix0 + ilag + i] - mub);
                    cross += (fsrc[ix0 + ilag + i] - mub) * (fref[ix0        + i] - mua);
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
            lag[iout] = ((float) irho2max) + cross;
            rho[iout] = mua * cross * cross + mub * cross + rhomax[1];
            sigref[iout] = sqrt(vara);
            sigsrc[iout] = sqrt(varb);
        }

        if (abs(rho[iout]) > 1) rho[iout] = (rho[iout] > 0) ? 1.0f : -1.0f;
    }

    __global__ 
    void calc_nxc_lagpairs(
        long long nref0,    // number of reference kernels to use
        long long * ref0s,  // starting indices of ref. kernels (len nref)
        long long lref,     // num. of samples in each kernel
        long long nsrc,     // num. of samples to search +/- iref
        long long nx,       // num. of samples in each signal
        long long nvec,     // num. of signals to compare
        const float * fref, // ref. signal to compare to (len nvec*nx)
        const float * fsrc, // search signal compared against (len nvec*nx)
        float * rho,        // output norm. xcorr coefs (len nvec*nref)
        float * lag,        // output lag estimates (len nvec*nref)
        float epsilon       // a non-negative value to bias denominator of NXCC calculation
    ) {
        long long tpb, tid, ivec, iref, ix0, iout;
        float rho2max, cross, vara, varb, mua, mub, flref;
        long long i, isrc0min, isrc0max, irho2max, ilag;
        
        // get cuda step sizes
        tpb = blockDim.x * blockDim.y * blockDim.z; // threads per block

        // Unique thread ID
        tid  = threadIdx.x + threadIdx.y * blockDim.x;
        tid += threadIdx.z * blockDim.x * blockDim.y;
        tid += tpb * (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);

        if (tid >= nvec * nref0) return;

        // get which signal pair and which reference kernel to assess in this function call
        ivec = tid / nref0; // which signal pair is being investigated
        iref = tid % nref0; // which reference kernel is being used

        ix0  = ivec*nx   + ref0s[iref]; // index of first sample of reference kernel
        iout = ivec*nref0 + iref;

        // determine the valid search bounds within this signal
        isrc0min = (ref0s[iref]-nsrc >= 0 ) ? -nsrc : -ref0s[iref];
        isrc0max = (ref0s[iref]+nsrc+lref < nx) ? nsrc : nx - ref0s[iref] - lref-1;

        rho2max = -1.0;
        irho2max = isrc0min;

        // calcualte the mean and STD of the reference kernel (once)
        flref = (float) lref; // float lref
        // mean loop
        mua = vara = 0.0f;
        for (i = 0; i < lref; ++i) mua += fref[ix0+i];
        mua /= flref;
        // variance loop
        for (i = 0; i < lref; ++i) vara += (fref[ix0+i]-mua) * (fref[ix0+i]-mua);
        vara /= flref;

        // pass through once to find the coordinates of the peak
        for (ilag = isrc0min; ilag < isrc0max+1; ++ilag) 
        {
            mub = varb = cross = 0.0f;

            // calculate search kernel mean
            for (i = 0; i < lref; ++i) mub += fsrc[ix0 + ilag + i];
            mub /= flref;

            // calcualte search kernel variance and search x reference covariance
            for (i = 0; i < lref; ++i) 
            {
                varb  += (fsrc[ix0 + ilag + i] - mub) * (fsrc[ix0 + ilag + i] - mub);
                cross += (fsrc[ix0 + ilag + i] - mub) * (fref[ix0        + i] - mua);
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
            rho   [iout] = 0.0f;
            lag   [iout] = 0.0f;
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
                for (i = 0; i < lref; ++i) mub += fsrc[ix0 + ilag + i];
                mub /= flref;

                // calcualte search kernel variance and search x reference covariance
                for (i = 0; i < lref; ++i) 
                {
                    varb  += (fsrc[ix0 + ilag + i] - mub) * (fsrc[ix0 + ilag + i] - mub);
                    cross += (fsrc[ix0 + ilag + i] - mub) * (fref[ix0        + i] - mua);
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
            lag[iout] = ((float) irho2max) + cross;
            rho[iout] = mua * cross * cross + mub * cross + rhomax[1];
        }

        if (abs(rho[iout]) > 1) rho[iout] = (rho[iout] > 0) ? 1.0f : -1.0f;
    }
}