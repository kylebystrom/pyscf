#include <stdio.h>
#include <math.h>
#include "config.h"

#define BLKSIZE 1024
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

void fast_SphBsli0(double * xs, int n, double * out)
{
    int nblk = n/BLKSIZE;

#pragma omp parallel
{
    int i;
#pragma omp for schedule(dynamic)
    for (i = 0; i < n; ++i) {
        out[i] = sinh(xs[i]) / xs[i];
    }
}
//     int i, i0, i1, iblk;
// #pragma omp parallel for schedule(dynamic)
//     for (iblk = 0; iblk < nblk; ++iblk) {
//         i0 = iblk*BLKSIZE;
//         i1 = MIN((iblk+1)*BLKSIZE,n);
//         for (i = i0; i < i1; ++i) {
//             out[i] = sinh(xs[i]) / xs[i];
//         }
//     }

}

void fast_SphBsli1(double * xs, int n, double * out)
{
#pragma omp parallel
{
    int i;
#pragma omp for schedule(dynamic)
    for (i = 0; i < n; ++i) {
        out[i] = (xs[i] * cosh(xs[i]) - sinh(xs[i])) / (xs[i] * xs[i]);
    }
}
}

void fast_SphBslin(double * xs, int n, int l, double * out)
// n: size of xs; l: order
{
    if (l == 0)
        fast_SphBsli0(xs, n, out);
    else if (l == 1)
        fast_SphBsli0(xs, n, out);
}
