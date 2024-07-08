NXCORR_PAIRWISE_W_VAR_SRC = r'''
extern "C"

__constant__ struct __NXCORR_PARAMS__ {
    // Shape of input
    int Ns;     // samples per signal
    int Np;     // number of signals

    // shape of results
    int Npp;    // number of point pairs being evaluated
    int Ncc;    // number of correlations per signal
    int Npm;    // number of indices to search up and down -> 2*Npm + 1

    // search parameters
    int t0;     // starting index for first reference kernel
    int dt;     // step between consecutive reference kernels
    int Nt;     // length of kernel in signal sample indices
} nxcpar;

__global__ void correlate(
        const float *sig,   // signal tensor [Np x Ns]
        const int *pref,    // refernce signal index [Npp]
        const int *pser,    // search signal index [Npp]
        float *covar,       // reference-search covariance [Npp x Ncc x (2*Npm + 1)]
        float *rvar,        // reference kernel variance [Npp x Ncc x (2*Npm + 1)]
        float *svar         // search kernel variance [Npp x Ncc x (2*Npm + 1)]
    ) 
{
    // calculate the point pair index and ensure it is within bounds
    int ippsr = blockIdx.x * gridDim.y + blockIdx.y;
    if (ippsr >= nxcpar.Npp) reutrn;
    int ipref = nxcpar.Ns * pref[ippsr];
    int ipser = nxcpar.Ns * pser[ippsr];

    // calculate the bounds of the reference kernel
    int t0kref = nxcpar.t0 + nxcpar.dt * blockIdx.z;
    int trmin = ipref + t0kref;
    int trmax = trmin + nxcpar.Nt;

    // calculate the bounds of the search kernel
    int dtsearch = threadIdx.x - nxcpar.Npm;
    int tsmin = ipser + t0kref + dtsearch;
    int tsmax = tsmin + nxcpar.Nt;

    // adjust kernel bounds to be within signal bounds
    if (trmin < ipref) {tsmin -= trmin + ipref; trmin = ipref;}
    if (tsmin < ipser) {trmin -= tsmin + ipser; tsmin = ipser;}
    if (trmax > ipref + nxcpar.Ns) {tsmax -= trmax - ipref - nxcpar.Ns; trmax = ipref + nxcpar.Ns;}
    if (tsmax > ipser + nxcpar.Ns) {trmax -= tsmax - ipser - nxcpar.Ns; tsmax = ipser + nxcpar.Ns;}
    int Nit = trmax - trmin;

    // calculate mean of each kernel
    float muref, muser = 0.0f;
    for(int it = 0; it < Nit; ++it)
    {
        muref += sig[ipref + trmin + it]; 
        muser += sig[ipser + tsmin + it];
    }
    muref /= nxcpar.Nt;
    muser /= nxcpar.Nt;

    // calculate covar and var for the kernels
    float cov, varr, vars, xs, xr = 0.0f;
    for(int it = 0; it < Nit; ++it)
    {
        xr = (sig[ipref + trmin + it] - muref);
        xs = (sig[ipser + trmin + it] - muser);
        cov  += xr * xs; 
        vars += xs * xs;
        varr += xr * xr;
    }
    cov  /= nxcpar.Nt;
    vars /= nxcpar.Nt;
    varr /= nxcpar.Nt;
    
    // save results
    ipref = (2*nxcpar.Npm + 1) * (nxcpar.Ncc * ippsr + blockIdx.z) + threadIdx.z;
    covar[ipref] = cov;
    svar[ipref] = vars;
    rvar[ipref] = varr;
}
'''