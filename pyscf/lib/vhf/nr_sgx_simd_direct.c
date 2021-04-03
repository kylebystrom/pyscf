/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "nr_direct.h"
#include "simd.h"

#include <immintrin.h>
#include <mm_malloc.h>

#define MAX(I,J)        ((I) > (J) ? (I) : (J))

typedef struct {
        int ncomp;
        int v_dims[3];
        double *data;
} SGXJKArrayM;

typedef struct {
        SGXJKArrayM *(*allocate)(int *shls_slice, int *ao_loc, int ncomp);
        void (*contract)(__MD *eri, double *dm, SGXJKArrayM *vjk,
                         int i0, int i1, int j0, int j1, int k0);
        void (*set0)(SGXJKArrayM *, int);
        void (*send)(SGXJKArrayM *, int, double *);
        void (*finalize)(SGXJKArrayM *, double *);
        void (*sanity_check)(int *shls_slice);
} SGXJKOperatorM;

int GTOmax_shell_dim(const int *ao_loc, const int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

#define DECLARE_ALL \
        const int *atm = envs->atm; \
        const int *bas = envs->bas; \
        const double *env = envs->env; \
        const int natm = envs->natm; \
        const int nbas = envs->nbas; \
        const int *ao_loc = envs->ao_loc; \
        const int *shls_slice = envs->shls_slice; \
        const CINTOpt *cintopt = envs->cintopt; \
        const int ish0 = shls_slice[0]; \
        const int ish1 = shls_slice[1]; \
        const int jsh0 = shls_slice[2]; \
        const int jsh1 = shls_slice[3]; \
        const int ksh0 = shls_slice[4]; \
        const int ioff = ao_loc[ish0]; \
        const int joff = ao_loc[jsh0]; \
        int i0, j0, i1, j1, ish, jsh, idm; \
        int shls[3]; \
        int (*fprescreen)(); \
        if (vhfopt) { \
                fprescreen = vhfopt->fprescreen; \
        } else { \
                fprescreen = CVHFnoscreen; \
        } \

/*
 * for given ksh, lsh, loop all ish, jsh
 */
void SGXdot_mm_nrs1(int (*intor)(), SGXJKOperatorM **jkop, SGXJKArrayM **vjk,
                    double **dms, double *buf, double *cache, int n_dm, int ksh,
                    CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;

        shls[2] = ksh0 + ksh;

        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh < jsh1; jsh++) {
                shls[0] = ish;
                shls[1] = jsh;
                if ((*fprescreen)(shls, vhfopt, atm, bas, env)
                    && (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                cintopt, cache)) {
                        i0 = ao_loc[ish  ] - ioff;
                        j0 = ao_loc[jsh  ] - joff;
                        i1 = ao_loc[ish+1] - ioff;
                        j1 = ao_loc[jsh+1] - joff;
                        for (idm = 0; idm < n_dm; idm++) {
                                jkop[idm]->contract(buf, dms[idm], vjk[idm],
                                                    i0, i1, j0, j1, ksh);
                        }
                }
        } }
}

/*
 * ish >= jsh
 */
void SGXdot_mm_nrs2(int (*intor)(), SGXJKOperatorM **jkop, SGXJKArrayM **vjk,
                   double **dms, double *buf, double *cache, int n_dm, int ksh,
                   CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;

        shls[2] = ksh0 + ksh;

        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh <= ish; jsh++) {
                shls[0] = ish;
                shls[1] = jsh;
                if ((*fprescreen)(shls, vhfopt, atm, bas, env)
                    && (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                cintopt, cache)) {
                        i0 = ao_loc[ish  ] - ioff;
                        j0 = ao_loc[jsh  ] - joff;
                        i1 = ao_loc[ish+1] - ioff;
                        j1 = ao_loc[jsh+1] - joff;
                        for (idm = 0; idm < n_dm; idm++) {
                                jkop[idm]->contract(buf, dms[idm], vjk[idm],
                                                    i0, i1, j0, j1, ksh);
                        }
                }
        } }
}

void double_to_simd_dm(double* out, double* in, int ng, int nao) {
    // k * nao + i
    // mk * nao * SIMDD + SIMDD * i + ik
    int nm = (ng - 1) / SIMDD + 1;
    int k, i, mk, ik;
    for (k = 0; k < nao; k++) {
        mk = k / SIMDD;
        ik = k % SIMDD;
        for (i = 0; i < nao; i++) {
            out[(mk * nao + i) * SIMDD + ik] = in[k * nao + i];
        }
    }
}

void simd_to_double_dm(double* out, double* in, int ng, int nao) {
    // k * nao + i
    // mk * nao * SIMDD + SIMDD * i + ik
    int nm = (ng - 1) / SIMDD + 1;
    int k, i, mk, ik;
    for (k = 0; k < nao; k++) {
        mk = k / SIMDD;
        ik = k % SIMDD;
        for (i = 0; i < nao; i++) {
            out[k * nao + i] = in[(mk * nao + i) * SIMDD + ik];
        }
    }
}

void set_double_vec_zero(double* x, int N) {
    int i;
    for (i = 0; i < N; i++) {
        x[i] = 0;
    }
}

void SGXnr_direct_simd_drv(int (*intor)(), void (*fdot)(), SGXJKOperatorM **jkop,
                           double **dms, double **vjk, int n_dm, int ncomp,
                           int *shls_slice, int *ao_loc,
                           CINTOpt *cintopt, CVHFOpt *vhfopt,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, ao_loc, NULL,
                cintopt, ncomp};

        const int ksh0 = shls_slice[4];
        const int ksh1 = shls_slice[5];
        int nksh = ksh1 - ksh0;
        int di = GTOmax_shell_dim(ao_loc, shls_slice, 2);
        int cache_size = GTOmax_cache_size(intor, shls_slice, 2,
                                           atm, natm, bas, nbas, env);

        int nao = ao_loc[nbas];
        int nmm = ((nksh - 1) / SIMDD + 1) * SIMDD;
        int dm_size = nao * nmm;

        double *new_dms[n_dm];
        double *backup_vjk[n_dm];
        for (int i = 0; i < n_dm; i++) {
                if (1) {
                        new_dms[i] = (double*) _mm_malloc(dm_size * sizeof(double),
                                                          SIMDD * sizeof(double));
                        backup_vjk[i] = vjk[i];
                        vjk[i] = (double*) _mm_malloc(dm_size * sizeof(double),
                                                      SIMDD * sizeof(double));
                        double_to_simd_dm(new_dms[i], dms[i], nksh, nao);
                } else {
                        new_dms[i] = dms[i];
                }
        }

#pragma omp parallel default(none) \
        shared(intor, fdot, jkop, ao_loc, shls_slice, \
               dms, vjk, n_dm, ncomp, nbas, vhfopt, envs, \
               nksh, di, cache_size, new_dms)
{
        int i, ksh;
        SGXJKArrayM *v_priv[n_dm];
        for (i = 0; i < n_dm; i++) {
                v_priv[i] = jkop[i]->allocate(shls_slice, ao_loc, ncomp);
        }
        // TODO check that these are correct size
        // TODO should these be aligned???
        double *buf = _mm_malloc(SIMDD*di*di*ncomp*sizeof(double),
                                 SIMDD*sizeof(double));
        set_double_vec_zero(buf, SIMDD*di*di*ncomp);
        double *cache = calloc(SIMDD*cache_size, sizeof(double));
#pragma omp for nowait schedule(dynamic, 1)
        for (ksh = 0; ksh < nksh; ksh+=SIMDD) {
                for (i = 0; i < n_dm; i++) {
                        jkop[i]->set0(v_priv[i], ksh);
                }
                (*fdot)(intor, jkop, v_priv, dms, buf, cache, n_dm, ksh,
                        vhfopt, &envs);
                for (i = 0; i < n_dm; i++) {
                        jkop[i]->send(v_priv[i], ksh, vjk[i]);
                }
        }
#pragma omp critical
{
        for (i = 0; i < n_dm; i++) {
                jkop[i]->finalize(v_priv[i], vjk[i]);
        }
}
        free(buf);
        free(cache);

}
        for (int i = 0; i < n_dm; i++) {
                if (1) { // TODO condition
                        _mm_free(new_dms[i]);
                        simd_to_double_dm(backup_vjk[i], vjk[i], nksh, nao);
                        _mm_free(vjk[i]);
                        vjk[i] = backup_vjk[i];
                }
        }
}

#define JTYPE1  1
#define JTYPE2  2
#define KTYPE1  3

#define ALLOCATE(label, task) \
        static SGXJKArrayM *SGXJKOperatorM_allocate_##label(int *shls_slice, int *ao_loc, int ncomp) \
{ \
        SGXJKArrayM *jkarray = malloc(sizeof(SGXJKArrayM)); \
        jkarray->v_dims[0]  = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]; \
        jkarray->v_dims[1]  = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]; \
        jkarray->v_dims[2]  = ao_loc[shls_slice[5]] - ao_loc[shls_slice[4]]; \
        if (task == JTYPE1) { \
                jkarray->data = _mm_malloc(SIMDD * sizeof(double) * ncomp, \
                                           SIMDD * sizeof(double)); \
        } else if (task == JTYPE2) { \
                jkarray->data = calloc(ncomp * jkarray->v_dims[0] * jkarray->v_dims[1], sizeof(double)); \
        } else { \
                jkarray->data = _mm_malloc(SIMDD * sizeof(double) * ncomp * jkarray->v_dims[0], \
                                           SIMDD * sizeof(double)); \
        } \
        jkarray->ncomp = ncomp; \
        return jkarray; \
} \
static void SGXJKOperatorM_set0_##label(SGXJKArrayM *jkarray, int k) \
{ \
        int ncomp = jkarray->ncomp; \
        int i; \
        double *data = jkarray->data; \
        if (task == JTYPE1) { \
                for (i = 0; i < ncomp * SIMDD; i++) { \
                        data[i] = 0; \
                } \
        } else if (task == KTYPE1) { \
                for (i = 0; i < ncomp * jkarray->v_dims[0] * SIMDD; i++) { \
                        data[i] = 0; \
                } \
        } \
} \
static void SGXJKOperatorM_send_##label(SGXJKArrayM *jkarray, int k, double *out) \
{ \
        int ncomp = jkarray->ncomp; \
        int i, icomp; \
        double *data = jkarray->data; \
        int ni = jkarray->v_dims[0]; \
        int nk = jkarray->v_dims[2]; \
        if (task == JTYPE1) { \
                for (i = 0; i < ncomp; i++) { \
                        out[i*nk+k] = data[i]; \
                } \
        } else if (task == KTYPE1) { \
                for (icomp = 0; icomp < ncomp; icomp++) { \
                        for (i = 0; i < ni; i++) { \
                                out[k*ni+i] = data[i]; \
                        } \
                        out += nk * ni; \
                        data += ni; \
                } \
        } \
} \
static void SGXJKOperatorM_final_##label(SGXJKArrayM *jkarray, double *out) \
{ \
        int i; \
        double *data = jkarray->data; \
        if (task == JTYPE2) { \
                for (i = 0; i < jkarray->ncomp * jkarray->v_dims[0] * jkarray->v_dims[1]; i++) { \
                        out[i] += data[i]; \
                } \
        } \
        SGXJKOperatorM_deallocate_simd(jkarray); \
}

#define ADD_OP(fname, task, type) \
        ALLOCATE(fname, task) \
SGXJKOperatorM SGX##fname = {SGXJKOperatorM_allocate_##fname, fname, \
        SGXJKOperatorM_set0_##fname, SGXJKOperatorM_send_##fname, \
        SGXJKOperatorM_final_##fname, \
        SGXJKOperatorM_sanity_check_##type##_simd}

static void SGXJKOperatorM_deallocate_simd(SGXJKArrayM *jkarray)
{
        free(jkarray->data);
        free(jkarray);
}

static void SGXJKOperatorM_sanity_check_s1_simd(int *shls_slice)
{
}
static void SGXJKOperatorM_sanity_check_s2_simd(int *shls_slice)
{
        if (!((shls_slice[0] == shls_slice[2]) &&
              (shls_slice[1] == shls_slice[3]))) {
                fprintf(stderr, "Fail at s2\n");
                exit(1);
        };
}

static void nrs1_ijg_ji_g_simd(__MD *eri, double *dm, SGXJKArrayM *out,
                               int i0, int i1, int j0, int j1, int k0)
{
        const int ncol = out->v_dims[0];
        int i, j, icomp;
        __MD g;
        double *data = out->data;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                g = MM_SET0();
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        g += eri[ij] * MM_SET1(dm[j*ncol+i]);
                } }
                MM_STORE(data+icomp*SIMDD, g);
        }
}
ADD_OP(nrs1_ijg_ji_g_simd, JTYPE1, s1);

static void nrs2_ijg_ji_g_simd(__MD *eri, double *dm, SGXJKArrayM *out,
                               int i0, int i1, int j0, int j1, int k0)
{
        if (i0 == j0) {
                return nrs1_ijg_ji_g_simd(eri, dm, out, i0, i1, j0, j1, k0);
        }

        const int ncol = out->v_dims[0];
        int i, j, icomp;
        __MD g;
        double *data = out->data;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                g = MM_SET0();
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        g += eri[ij] * MM_SET1(dm[j*ncol+i] + dm[i*ncol+j]);
                } }
                MM_STORE(data+icomp*SIMDD, g);
        }
}
ADD_OP(nrs2_ijg_ji_g_simd, JTYPE1, s2);

static void nrs1_ijg_g_ij_simd(__MD *eri, double *dm, SGXJKArrayM *out,
                               int i0, int i1, int j0, int j1, int k0)
{
        int ni = out->v_dims[0];
        int nj = out->v_dims[1];
        int i, j, icomp;
        int l;
        double *data = out->data;
        ALIGNMM double tmp[SIMDD];

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        MM_STORE(tmp, eri[ij] * MM_LOAD(dm+k0*SIMDD));
                        for (l = 0; l < SIMDD; l++) {
                            data[i*nj+j] += tmp[l];
                        }
                } }
                data += ni * nj;
        }
}
ADD_OP(nrs1_ijg_g_ij_simd, JTYPE2, s1);

SGXJKOperatorM SGXnrs2_ijg_g_ij_simd = {SGXJKOperatorM_allocate_nrs1_ijg_g_ij_simd,
        nrs1_ijg_g_ij_simd, SGXJKOperatorM_set0_nrs1_ijg_g_ij_simd,
        SGXJKOperatorM_send_nrs1_ijg_g_ij_simd, SGXJKOperatorM_final_nrs1_ijg_g_ij_simd,
        SGXJKOperatorM_sanity_check_s2_simd};

static void nrs1_ijg_gj_gi_simd(__MD *eri, double *dm, SGXJKArrayM *out,
                                int i0, int i1, int j0, int j1, int k0)
{
        const int ncol = out->v_dims[1];
        double *data = out->data;
        int i, j, icomp;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        //data[i] += eri[ij] * MM_LOAD(dm+(k0*ncol+j])*SIMDD);
MM_STORE(data+i*SIMDD, MM_FMA(eri[ij], MM_LOAD(dm+(k0*ncol+j)*SIMDD), MM_LOAD(data+i*SIMDD)));
                } }
                //data += out->v_dims[0];
                data += out->v_dims[0] * SIMDD;
        }
}
ADD_OP(nrs1_ijg_gj_gi_simd, KTYPE1, s1);

static void nrs2_ijg_gj_gi_simd(__MD *eri, double *dm, SGXJKArrayM *out,
                                int i0, int i1, int j0, int j1, int k0)
{
        if (i0 == j0) {
                return nrs1_ijg_gj_gi_simd(eri, dm, out, i0, i1, j0, j1, k0);
        }

        const int ncol = out->v_dims[0];
        double *data = out->data;
        int i, j, icomp;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                        //data[i] += eri[ij] * dm[k0*ncol+j];
                        //data[j] += eri[ij] * dm[k0*ncol+i];
MM_STORE(data+i*SIMDD, MM_FMA(eri[ij], MM_LOAD(dm+(k0*ncol+j)*SIMDD), MM_LOAD(data+i*SIMDD)));
MM_STORE(data+j*SIMDD, MM_FMA(eri[ij], MM_LOAD(dm+(k0*ncol+i)*SIMDD), MM_LOAD(data+j*SIMDD)));
                } }
                //data += out->v_dims[0];
                data += out->v_dims[0] * SIMDD;
        }
}
ADD_OP(nrs2_ijg_gj_gi_simd, KTYPE1, s2);
