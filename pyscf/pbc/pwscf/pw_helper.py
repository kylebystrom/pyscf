""" Helper functions for PW SCF
"""


import time
import copy
import h5py
import tempfile
import numpy as np
import scipy.linalg

from pyscf.pbc import tools, df
from pyscf import lib
from pyscf.lib import logger


""" Helper functions
"""
def get_kcomp(C_ks, k, load=True, occ=None):
    if C_ks is None: return None
    if isinstance(C_ks, list):
        if occ is None:
            return C_ks[k]
        else:
            return C_ks[k][occ]
    else:
        key = "%d"%k
        if load:
            if occ is None:
                return C_ks[key][()]
            else:
                if isinstance(occ, np.ndarray):
                    occ = occ.tolist()
                return C_ks[key][occ]
        else:
            return C_ks[key]
def safe_write(h5grp, key, val, occ=None):
    if key in h5grp:
        if occ is None:
            if h5grp[key].shape == val.shape:
                h5grp[key][()] = val
            else:
                del h5grp[key]
                h5grp[key] = val
        else:
            h5grp[key][occ] = val
    else:
        h5grp[key] = val
def set_kcomp(C_k, C_ks, k, occ=None):
    if isinstance(C_ks, list):
        if occ is None:
            C_ks[k] = C_k
        else:
            C_ks[k][occ] = C_k
    else:
        key = "%d"%k
        safe_write(C_ks, key, C_k, occ)
def acc_kcomp(C_k, C_ks, k, occ=None):
    if isinstance(C_ks, list):
        if occ is None:
            C_ks[k] += C_k
        else:
            C_ks[k][occ] += C_k
    else:
        key = "%d"%k
        if occ is None:
            C_ks[key][()] += C_k
        else:
            if isinstance(occ, np.ndarray):
                occ = occ.tolist()
            C_ks[key][occ] += C_k
def scale_kcomp(C_ks, k, scale):
    if isinstance(C_ks, list):
        C_ks[k] *= scale
    else:
        key = "%d"%k
        C_ks[key][()] *= scale


def timing_call(func, args, tdict, tname):
    tick = np.asarray([time.clock(), time.time()])

    res = func(*args)

    tock = np.asarray([time.clock(), time.time()])
    if not tname in tdict:
        tdict[tname] = np.zeros(2)
    tdict[tname] += tock - tick

    return res


def orth(cell, C, thr_nonorth=1e-6, thr_lindep=1e-12, follow=True):
    n = C.shape[0]
    S = lib.dot(C.conj(), C.T)
    nonorth_err = np.max(np.abs(S - np.eye(S.shape[0])))
    if nonorth_err < thr_nonorth:
        return C

    e, u = scipy.linalg.eigh(S)
    idx_keep = np.where(e > thr_lindep)[0]
    nkeep = idx_keep.size
    if n == nkeep:  # symm orth
        if follow:
            # reorder to maximally overlap original orbs
            idx = []
            for i in range(n):
                order = np.argsort(np.abs(u[i]))[::-1]
                for j in order:
                    if not j in idx:
                        break
                idx.append(j)
            U = lib.dot(u[:,idx]*e[idx]**-0.5, u[:,idx].conj()).T
        else:
            U = lib.dot(u*e**-0.5, u.conj()).T
    else:   # cano orth
        U = (u[:,idx_keep]*e[idx_keep]**-0.5).T
    C = lib.dot(U, C)

    return C


def get_nocc_ks_from_mocc(mocc_ks):
    return np.asarray([np.sum(np.asarray(mocc) > 0) for mocc in mocc_ks])


def get_C_ks_G(cell, kpts, mo_coeff_ks, n_ks, out=None, verbose=0):
    """ Return Cik(G) for input MO coeff. The normalization convention is such that Cik(G).conj()@Cjk(G) = delta_ij.
    """
    log = logger.new_logger(cell, verbose)

    nkpts = len(kpts)
    if out is None: out = [None] * nkpts

    dtype = np.complex128
    dsize = 16

    mydf = df.FFTDF(cell)
    mesh = mydf.mesh
    ni = mydf._numint

    coords = mydf.grids.coords
    ngrids = coords.shape[0]
    weight = mydf.grids.weights[0]
    fac = (weight/ngrids)**0.5

    frac = 0.5  # to be safe
    cur_memory = lib.current_memory()[0]
    max_memory = (cell.max_memory - cur_memory) * frac
    log.debug1("max_memory= %s MB (currently used %s MB)", cell.max_memory, cur_memory)
    # FFT needs 2 temp copies of MOs
    extra_memory = 2*ngrids*np.max(n_ks)*dsize / 1.e6
    # add 1 for ao_ks
    perk_memory = ngrids*(np.max(n_ks)+1)*dsize / 1.e6
    kblksize = min(int(np.floor((max_memory-extra_memory) / perk_memory)),
                   nkpts)
    if kblksize <= 0:
        log.warn("Available memory %s MB cannot perform conversion for orbitals of a single k-point. Calculations may crash and `cell.memory = %s` is recommended.", max_memory, (perk_memory + extra_memory) / frac + cur_memory)

    log.debug1("max memory= %s MB, extra memory= %s MB, perk memory= %s MB, kblksize= %s", max_memory, extra_memory, perk_memory, kblksize)

    for k0,k1 in lib.prange(0, nkpts, kblksize):
        nk = k1 - k0
        C_ks_R = [np.zeros([ngrids,n_ks[k]], dtype=dtype)
                   for k in range(k0,k1)]
        for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts[k0:k1]):
            ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
            for krel, ao in enumerate(ao_ks):
                k = krel + k0
                kpt = kpts[k].reshape(-1,1)
                C_k = mo_coeff_ks[k][:,:n_ks[k]]
                C_ks_R[krel][p0:p1] = lib.dot(ao, C_k)
                if k > 0:
                    C_ks_R[krel][p0:p1] = np.exp(-1j * lib.dot(coords[p0:p1],
                        kpt)) * lib.dot(ao, C_k)
            ao = ao_ks = None

        for krel in range(nk):
            C_k_R = tools.fft(C_ks_R[krel].T * fac, mesh)
            set_kcomp(C_k_R, out, krel+k0)

    return out


""" kinetic energy
"""
def apply_kin_kpt(C_k, kpt, mesh, Gv):
    no = C_k.shape[0]
    kG = kpt + Gv if np.sum(np.abs(kpt)) > 1.E-9 else Gv
    kG2 = np.einsum("gj,gj->g", kG, kG) * 0.5
    Cbar_k = C_k * kG2

    return Cbar_k


""" Charge mixing methods
"""
class SimpleMixing:
    def __init__(self, mf, beta=0.3):
        self.beta = beta
        self.cycle = 0

    def next_step(self, mf, f, ferr):
        self.cycle += 1

        return f - ferr * self.beta

from pyscf.lib.diis import DIIS
class AndersonMixing:
    def __init__(self, mf, ndiis=10, diis_start=1):
        self.diis = DIIS()
        self.diis.space = ndiis
        self.diis.min_space = diis_start
        self.cycle = 0

    def next_step(self, mf, f, ferr):
        self.cycle += 1

        return self.diis.update(f, ferr)
