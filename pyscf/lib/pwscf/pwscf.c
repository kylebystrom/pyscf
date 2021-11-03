#include <stdio.h>
#include <math.h>
#include "config.h"


void fast_SphBsli0(double * xs, int n, double * out)
{
#pragma omp parallel
{
    int i;
    double x;
#pragma omp for schedule(static)
    for (i = 0; i < n; ++i) {
        x = xs[i];
        out[i] = sinh(x) / x;
    }
}
}

void fast_SphBsli1(double * xs, int n, double * out)
{
#pragma omp parallel
{
    int i;
    double x;
#pragma omp for schedule(static)
    for (i = 0; i < n; ++i) {
        x = xs[i];
        out[i] = (x*cosh(x) - sinh(x)) / (x*x);
    }
}
}

void fast_SphBsli2(double * xs, int n, double * out)
{
#pragma omp parallel
{
    int i;
    double x;
#pragma omp for schedule(static)
    for (i = 0; i < n; ++i) {
        x = xs[i];
        out[i] = ((x*x+3.)*sinh(x) - 3.*x*cosh(x)) / (x*x*x);
    }
}
}

void fast_SphBsli3(double * xs, int n, double * out)
{
#pragma omp parallel
{
    int i;
    double x;
#pragma omp for schedule(static)
    for (i = 0; i < n; ++i) {
        x = xs[i];
        out[i] = ((x*x*x+15.*x)*cosh(x) -
                (6.*x*x+15.)*sinh(x)) / (x*x*x*x);
    }
}
}

void fast_SphBslin(double * xs, int n, int l, double * out)
// n: size of xs; l: order
{
    if (l == 0)
        fast_SphBsli0(xs, n, out);
    else if (l == 1)
        fast_SphBsli1(xs, n, out);
    else if (l == 2)
        fast_SphBsli2(xs, n, out);
    else if (l == 3)
        fast_SphBsli3(xs, n, out);
}
