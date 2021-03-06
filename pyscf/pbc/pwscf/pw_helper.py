""" Helper functions for PW SCF
"""


import time
import copy
import h5py
import tempfile
import numpy as np
import scipy.linalg

from pyscf.pbc import tools, df
from pyscf.pbc.gto import pseudo
from pyscf import lib
from pyscf.lib import logger


""" Helper functions
"""
def timing_call(func, args, tdict, tname):
    tick = np.asarray([time.clock(), time.time()])

    res = func(*args)

    tock = np.asarray([time.clock(), time.time()])
    if not tname in tdict:
        tdict[tname] = np.zeros(2)
    tdict[tname] += tock - tick

    return res


def get_no_ks_from_mocc(mocc_ks):
    return np.asarray([np.sum(np.asarray(mocc) > 0) for mocc in mocc_ks])


def get_C_ks_G(cell, kpts, mo_coeff_ks, n_ks, fC_ks=None, verbose=0):
    """ Return Cik(G) for input MO coeff. The normalization convention is such that Cik(G).conj()@Cjk(G) = delta_ij.
    """
    log = logger.new_logger(cell, verbose)

    nkpts = len(kpts)

    if not fC_ks is None:
        assert(isinstance(fC_ks, h5py.Group))
        C_ks_G = fC_ks
        incore = False
    else:
        C_ks_G = [None] * nkpts
        incore = True

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

        if incore:
            for krel in range(nk):
                C_k_R = C_ks_R[krel].T * fac
                C_ks_G[krel+k0] = tools.fft(C_k_R, mesh)
        else:
            for krel in range(nk):
                C_k_R = C_ks_R[krel].T * fac
                C_ks_G["%d"%(krel+k0)] = tools.fft(C_k_R, mesh)

    return C_ks_G


""" kinetic energy
"""
def apply_kin_kpt(C_k, kpt, mesh, Gv):
    no = C_k.shape[0]
    kG = kpt + Gv if np.sum(np.abs(kpt)) > 1.E-9 else Gv
    kG2 = np.einsum("gj,gj->g", kG, kG) * 0.5
    Cbar_k = C_k * kG2

    return Cbar_k


""" Pseudopotential
"""
def get_vpplocR(cell, mesh=None, Gv=None):
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    ngrids = Gv.shape[0]
    fac = ngrids / cell.vol
    SI = cell.get_SI()
    if cell.pseudo is None:
        Zs = cell.atom_charges()
        coulG = tools.get_coulG(cell, Gv=Gv)
        vpplocG = -np.einsum("a,ag,g->g", Zs, SI, coulG)
        vpplocR = tools.ifft(vpplocG, mesh).real * fac
    else:
        vpplocG = pseudo.get_vlocG(cell, Gv)
        vpplocG = -np.einsum('ij,ij->j', SI, vpplocG)
        vpplocR = tools.ifft(vpplocG, mesh).real * fac

    return vpplocR


def apply_ppl_kpt(C_k, mesh, vpplocR):

    Cbar_k = tools.fft(tools.ifft(C_k, mesh) * vpplocR, mesh)

    return Cbar_k


def apply_ppnl_kpt(cell, C_k, kpt, mesh, Gv):
    if cell.pseudo is None:
        return np.zeros_like(C_k)

    SI = cell.get_SI()
    no = C_k.shape[0]
    ngrids = Gv.shape[0]

    # non-local pp
    from pyscf import gto
    fakemol = gto.Mole()
    fakemol._atm = np.zeros((1,gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    buf = np.empty((48,ngrids), dtype=np.complex128)
    def get_Cbar_k_nl(kpt, C_k_):
        Cbar_k = np.zeros_like(C_k_)

        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)
        vppnl = 0
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                    pYlm_part = fakemol.eval_gto('GTOval', Gk)

                    p0, p1 = p1, p1+nl*(l*2+1)
                    # pYlm is real, SI[ia] is complex
                    pYlm = np.ndarray((nl,l*2+1,ngrids), dtype=np.complex128, buffer=buf[p0:p1])
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    #:SPG_lmi = np.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    #:SPG_lm_aoG = np.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    #:tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    #:vppnl += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
            if p1 > 0:
                SPG_lmi = buf[:p1]
                SPG_lmi *= SI[ia].conj()
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = np.asarray(hl)
                        SPG_lmi_ = SPG_lmi[p0:p1].reshape(nl,l*2+1,-1)
                        tmp = np.einsum("imG,IG->Iim", SPG_lmi_, C_k_)
                        tmp = np.einsum("ij,Iim->Ijm", hl, tmp)
                        Cbar_k += np.einsum("Iim,imG->IG", tmp, SPG_lmi_.conj())
        return Cbar_k / cell.vol

    Cbar_k = get_Cbar_k_nl(kpt, C_k)

    return Cbar_k


""" Columb energy
"""
def get_vj_R(cell, C_ks, mocc_ks, mesh=None, Gv=None):
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    nkpts = len(C_ks)
    ngrids = np.prod(mesh)
    no_ks = get_no_ks_from_mocc(mocc_ks)

    incore = isinstance(C_ks, list)

    vj_G = np.zeros(ngrids, dtype=np.complex128)
    for k in range(nkpts):
        Co_k = C_ks[k][:no_ks[k]] if incore else C_ks["%d"%k][:no_ks[k]]
        Co_k_R = tools.ifft(Co_k, mesh)
        vj_G += np.einsum("ig,ig->g", Co_k_R.conj(), Co_k_R)
    vj_G = tools.fft(vj_G, mesh)
    vj_G *= ngrids**2 / (cell.vol*nkpts)
    vj_G *= tools.get_coulG(cell, Gv=Gv)
    vj_R = tools.ifft(vj_G, mesh)

    return vj_R


def apply_vj_kpt(C_k, mesh, vj_R):
    Cbar_k = tools.fft(tools.ifft(C_k, mesh) * vj_R, mesh)

    return Cbar_k


""" Exchange energy
"""
def apply_vk_kpt(cell, C_k, kpt1, C_ks, mocc_ks, kpts,
                  mesh=None, Gv=None, exxdiv=None):
    r""" Apply the EXX operator to given MOs

    Math:
        Cbar_k(G) = \sum_{j,k'} \sum_{G'} rho_{jk',ik}(G') v(k-k'+G') C_k(G-G')
    Code:
        rho_r = C_ik_r * C_jk'_r.conj()
        rho_G = FFT(rho_r)
        coulG = get_coulG(k-k')
        v_r = iFFT(rho_G * coulG)
        Cbar_ik_G = FFT(v_r * C_jk'_r)
    """
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    ngrids = Gv.shape[0]
    nkpts = len(kpts)
    no_ks = get_no_ks_from_mocc(mocc_ks)
    fac = ngrids**2./(cell.vol*nkpts)

    incore = isinstance(C_ks, list)

    Cbar_k = np.zeros_like(C_k)
    C_k_R = tools.ifft(C_k, mesh)

    for k2 in range(nkpts):
        kpt2 = kpts[k2]
        coulG = tools.get_coulG(cell, kpt1-kpt2, exx=False, mesh=mesh)

        Co_k2 = C_ks[k2][:no_ks[k2]] if incore else C_ks["%d"%k2][:no_ks[k2]]
        Co_k2_R = tools.ifft(Co_k2, mesh)
        Co_k2 = None
        for j in range(no_ks[k2]):
            Cj_k2_R = Co_k2_R[j]
            vij_R = tools.ifft(
                tools.fft(C_k_R * Cj_k2_R.conj(), mesh) * coulG, mesh)
            Cbar_k += vij_R * Cj_k2_R

    Cbar_k = tools.fft(Cbar_k, mesh) * fac

    return Cbar_k


def apply_vk_kpt_ace(C_k, ace_xi_k):
    Cbar_k = lib.dot(lib.dot(C_k, ace_xi_k.conj().T), ace_xi_k)
    return Cbar_k


def apply_vk_s1(cell, C_ks, Ct_ks, mocc_ks, kpts, mesh, Gv,
                fout=None, dataname="Cbar_ks"):
    """
        Args:
            fout (None or h5py.File object):
                Where to store the result vectors. If h5py.File object, results are written to it (overwritten dataname if exists). If None, incore mode is used and a list of Nk numpy arrays is returned.
    """
    nkpts = len(kpts)
    ngrids = np.prod(mesh)
    fac = ngrids**2./(cell.vol*nkpts)

    C_incore = isinstance(C_ks, list)
    Ct_incore = isinstance(Ct_ks, list)

    no_ks = [np.sum(mocc_ks[k]>0) for k in range(nkpts)]
    if Ct_incore:
        nt_ks = [Ct_ks[k].shape[0] for k in range(nkpts)]
    else:
        nt_ks = [Ct_ks["%d"%k].shape[0] for k in range(nkpts)]

    dtype = np.complex128
    dsize = 16

    if fout is None:
        # check if the output Ctbar_ks fits memory
        max_memory = (cell.max_memory - lib.current_memory()[0]) * 0.4
        est_memory = np.sum(nt_ks)*ngrids*dsize/1024**2.
        outcore = est_memory > max_memory
        if outcore:
            logger.warn(cell, "Result vectors cannot fit memory. Try outcore mode by specifying fout.")
            raise RuntimeError
        Cbar_ks = [None] * nkpts
    else:
        outcore = True
        if dataname in fout: del fout[dataname]
        Cbar_ks = fout.create_group(dataname)

    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fswap = lib.H5TmpFile(swapfile.name)
    swapfile = None

    if C_incore:
        for k in range(nkpts):
            fswap["Co_ks_R/%s"%k] = tools.ifft(C_ks[k][:no_ks[k]], mesh)
    else:
        for k in range(nkpts):
            fswap["Co_ks_R/%s"%k] = tools.ifft(C_ks["%d"%k][:no_ks[k]], mesh)
    if Ct_incore:
        for k in range(nkpts):
            fswap["Ct_ks_R/%s"%k] = tools.ifft(Ct_ks[k], mesh)
    else:
        for k in range(nkpts):
            fswap["Ct_ks_R/%s"%k] = tools.ifft(Ct_ks["%d"%k][()], mesh)

    for k1,kpt1 in enumerate(kpts):
        Ct_k1_R = fswap["Ct_ks_R/%d"%k1][()]
        Ctbar_k1 = np.zeros_like(Ct_k1_R)
        for k2,kpt2 in enumerate(kpts):
            coulG = tools.get_coulG(cell, kpt1-kpt2, exx=False, mesh=mesh)
            Co_k2_R = fswap["Co_ks_R/%d"%k2][()]
            for j in range(no_ks[k2]):
                Cj_k2_R = Co_k2_R[j]
                vij_R = tools.ifft(tools.fft(Ct_k1_R * Cj_k2_R.conj(), mesh) *
                                   coulG, mesh)
                Ctbar_k1 += vij_R * Cj_k2_R

        if outcore:
            Cbar_ks["%d"%k1] = tools.fft(Ctbar_k1, mesh) * fac
        else:
            Cbar_ks[k1] = tools.fft(Ctbar_k1, mesh) * fac
        Ctbar_k1 = None

    return Cbar_ks


def apply_vk_s2(cell, C_ks, mocc_ks, kpts, mesh, Gv,
                fout=None, dataname="Cbar_ks"):
    nkpts = len(kpts)
    ngrids = np.prod(mesh)
    fac = ngrids**2./(cell.vol*nkpts)

    C_incore = isinstance(C_ks, list)

    if C_incore:
        n_ks = [C_ks[k].shape[0] for k in range(nkpts)]
    else:
        n_ks = [C_ks["%d"%k].shape[0] for k in range(nkpts)]
    no_ks = [np.sum(mocc_ks[k]>0) for k in range(nkpts)]

    n_max = np.max(n_ks)
    no_max = np.max(no_ks)
    dtype = np.complex128
    dsize = 16

    if fout is None:
        # check if the output Cbar_ks fits memory
        max_memory = (cell.max_memory - lib.current_memory()[0]) * 0.4
        est_memory = np.sum(n_ks)*ngrids*dsize/1024**2.
        outcore = est_memory > max_memory
        if outcore:
            logger.warn(cell, "Result vectors cannot fit into memory. Try outcore mode by specifying fout.")
            raise RuntimeError
        Cbar_ks = [None] * nkpts
    else:
        outcore = True
        if dataname in fout: del fout[dataname]
        Cbar_ks = fout.create_group(dataname)

    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fswap = lib.H5TmpFile(swapfile.name)
    swapfile = None

    if C_incore:
        for k in range(nkpts):
            fswap["C_ks_R/%d"%k] = tools.ifft(C_ks[k], mesh)
    else:
        for k in range(nkpts):
            fswap["C_ks_R/%d"%k] = tools.ifft(C_ks["%d"%k][()], mesh)

    for k in range(nkpts):
        key = "%d"%k if outcore else k
        Cbar_ks[key] = np.zeros((n_ks[k],ngrids), dtype=dtype)

    buf1 = np.empty(n_max*ngrids, dtype=dtype)
    buf2 = np.empty(no_max*ngrids, dtype=dtype)
    for k1,kpt1 in enumerate(kpts):
        C_k1_R = fswap["C_ks_R/%d"%k1][()]
        no_k1 = no_ks[k1]
        n_k1 = n_ks[k1]
        Cbar_k1 = np.ndarray((n_k1,ngrids), dtype=dtype, buffer=buf1)
        Cbar_k1.fill(0)
        for k2,kpt2 in enumerate(kpts):
            if n_k1 == no_k1 and k2 > k1: continue

            C_k2_R = fswap["C_ks_R/%d"%k2][()]
            no_k2 = no_ks[k2]

            coulG = tools.get_coulG(cell, kpt1-kpt2, exx=False, mesh=mesh)

            # o --> o
            if k2 <= k1:
                Cbar_k2 = np.ndarray((no_k2,ngrids), dtype=dtype, buffer=buf2)
                Cbar_k2.fill(0)

                for i in range(no_k1):
                    jmax = i+1 if k2 == k1 else no_k2
                    jmax2 = jmax-1 if k2 == k1 else jmax
                    vji_R = tools.ifft(tools.fft(C_k2_R[:jmax].conj() *
                                       C_k1_R[i], mesh) * coulG, mesh)
                    Cbar_k1[i] += np.sum(vji_R * C_k2_R[:jmax], axis=0)
                    Cbar_k2[:jmax2] += vji_R[:jmax2].conj() * C_k1_R[i]

                if outcore:
                    Cbar_ks["%d"%k2][:no_k2] += Cbar_k2
                else:
                    Cbar_ks[k2][:no_k2] += Cbar_k2

            # o --> v
            if n_k1 > no_k1:
                for j in range(no_ks[k2]):
                    vij_R = tools.ifft(tools.fft(C_k1_R[no_k1:] *
                                                 C_k2_R[j].conj(), mesh) *
                                       coulG, mesh)
                    Cbar_k1[no_k1:] += vij_R  * C_k2_R[j]

        if outcore:
            Cbar_ks["%d"%k1][()] += Cbar_k1
        else:
            Cbar_ks[k1] += Cbar_k1

    if outcore:
        for k in range(nkpts):
            Cbar_ks["%d"%k][()] = tools.fft(Cbar_ks["%d"%k][()], mesh) * fac
    else:
        for k in range(nkpts):
            Cbar_ks[k] = tools.fft(Cbar_ks[k], mesh) * fac

    return Cbar_ks


def initialize_ACE_from_W(C_ks, W_ks, ace_xi_ks):

    C_incore = isinstance(C_ks, list)
    incore = isinstance(W_ks, list)

    nkpts = len(C_ks)
    for k in range(nkpts):
        C_k = C_ks[k] if C_incore else C_ks["%d"%k][()]
        W_k = W_ks[k] if incore else W_ks["%d"%k][()]
        B_k = lib.dot(C_k.conj(),W_k.T)
        if np.sum(np.abs(B_k)) < 1e-10:
            L_k = np.zeros_like(C_k)
        else:
            L_k = scipy.linalg.cholesky(B_k, lower=True)
            L_k = scipy.linalg.solve_triangular(L_k.conj(), W_k, lower=True)

        key = "%d"%k
        if key in ace_xi_ks: del ace_xi_ks[key]
        ace_xi_ks[key] = L_k

        # debug
        # xi_k = ace_xi_ks["%d"%k][()]
        # W_k_prime = (C_k @ xi_k.conj().T) @ xi_k
        # assert(np.linalg.norm(W_k - W_k_prime) < 1e-8)

    return ace_xi_ks


def initialize_ACE(cell, C_ks, mocc_ks, kpts, mesh, Gv, ace_xi_ks, Ct_ks=None):
    """
        Args:
            ace_xi_ks: h5py Group
    """

    if not ace_xi_ks is None:
        swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        fout = lib.H5TmpFile(swapfile.name)
        swapfile = None

        if Ct_ks is None:
            W_ks = apply_vk_s2(cell, C_ks, mocc_ks, kpts, mesh, Gv, fout=fout)
            # debug
            # nkpts = len(kpts)
            # W_ks_ = [apply_vk_kpt(cell, C_ks[k], kpts[k], C_ks, mocc_ks, kpts,
            #                        mesh=mesh, Gv=Gv, exxdiv=exxdiv)
            #                        for k in range(nkpts)]
            # for k in range(nkpts):
            #     print(np.linalg.norm(W_ks[k] - W_ks_[k]))
            # sys.exit(1)
            initialize_ACE_from_W(C_ks, W_ks, ace_xi_ks)
        else:
            W_ks = apply_vk_s1(cell, C_ks, Ct_ks, mocc_ks, kpts, mesh, Gv,
                               fout=fout)
            # debug
            # nkpts = len(kpts)
            # W_ks_ = [apply_vk_kpt(cell, C_ks[k], kpts[k], C_ks, mocc_ks, kpts,
            #                        mesh=mesh, Gv=Gv, exxdiv=exxdiv)
            #                        for k in range(nkpts)]
            # for k in range(nkpts):
            #     print(np.linalg.norm(W_ks[k] - W_ks_[k]))
            # sys.exit(1)
            initialize_ACE_from_W(Ct_ks, W_ks, ace_xi_ks)


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
