#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "cint_funcs.h"
#include "nr_direct.h"
#include "simd.h"

#include <immintrin.h>
#include <mm_malloc.h>

#define MAX(I,J)        ((I) > (J) ? (I) : (J))
#define MIN(I,J)        ((I) < (J) ? (I) : (J))

void test_simd_intor(int (*intor1)(), int (*intor2)(), int *shls, int *ao_loc,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
    int ncomp = 1;

    int ksh;
    int shls_slice[6];
    shls_slice[0] = shls[0];
    shls_slice[1] = shls[0]+1;
    shls_slice[2] = shls[1];
    shls_slice[3] = shls[1]+1;
    shls_slice[4] = shls[2];
    shls_slice[5] = shls[2]+1;
    int i, j;
    int i0 = shls_slice[0];
    int i1 = shls_slice[1];
    int j0 = shls_slice[2];
    int j1 = shls_slice[3];

    int di = GTOmax_shell_dim(ao_loc, shls_slice, 2);
    int cache_size = GTOmax_cache_size(intor1, shls_slice, 2,
                                       atm, natm, bas, nbas, env);

    double *buf_simd = _mm_malloc(SIMDD*di*di*ncomp*sizeof(double),
                           SIMDD*sizeof(double));
    set_double_vec_zero((double*) buf_simd, SIMDD*di*di*ncomp);
    double *cache_simd = calloc(SIMDD*cache_size, sizeof(double));
    intor2(buf_simd, NULL, shls, atm, natm, bas, nbas, env, NULL, cache_simd);

    double *buf = calloc(di*di*ncomp*SIMDD, sizeof(double));
    double *cache = calloc(cache_size, sizeof(double));
    for (ksh=0; ksh<SIMDD; ksh++) {
        intor1(buf, NULL, shls,
               atm, natm, bas, nbas, env, NULL, cache);
        for (j=j0; j<j1; j++) {
            for (i=i0; i<i1; i++) {
                printf("%e %e %e\n", buf[j*(i1-i0)+i], buf_simd[(j*(i1-i0)+i)*SIMDD+ksh],
                                     buf[j*(i1-i0)+i] -buf_simd[(j*(i1-i0)+i)*SIMDD+ksh]);
            }
        }
        shls[2]++;
    }
    for (ksh=0; ksh<SIMDD*di*di*ncomp; ksh++) {
        printf("%e\n", buf_simd[ksh]);
    }
}
