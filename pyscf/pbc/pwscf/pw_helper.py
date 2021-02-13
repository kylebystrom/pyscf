""" Helper functions for PW SCF
"""


import copy
import numpy as np
import scipy.linalg

from pyscf.pbc import tools, df
from pyscf.pbc.gto import pseudo
from pyscf import lib


def get_no_ks(mo_occ_ks):
    return np.sum(np.asarray(mo_occ_ks),axis=1).astype(int)//2


def get_Co_ks_G(cell, kpts, mo_coeff_ks, mo_occ_ks, mesh=None):
    """ Return Cik(G) for input MO coeff. The normalization convention is such that Cik(G).conj()@Cjk(G) = delta_ij.
    """
    nkpts = len(kpts)

    pcell = copy.copy(cell)
    if not mesh is None:
        pcell.mesh = mesh
        pcell.build()
    mydf = df.FFTDF(pcell)

    mesh = mydf.mesh
    ni = mydf._numint

    coords = mydf.grids.coords
    ngrids = coords.shape[0]
    weight = mydf.grids.weights[0]

    no_ks = get_no_ks(mo_occ_ks)
    Co_ks_R = [np.zeros([ngrids,no_ks[ik]], dtype=mo_coeff_ks[0].dtype)
        for ik in range(nkpts)]
    for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts):
        ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
        for k, ao in enumerate(ao_ks):
            Co_k = mo_coeff_ks[k][:,:no_ks[k]]
            Co_ks_R[k][p0:p1] = lib.dot(ao, Co_k)
            if k > 0:
                Co_ks_R[k][p0:p1] = np.exp(-1j * (coords[p0:p1] @
                    kpts[k].T)).reshape(-1,1) * lib.dot(ao, Co_k)
        ao = ao_ks = None

    for ik in range(nkpts):
        Co_ks_R[ik] *= weight**0.5

    Co_ks_G = [np.empty([no_ks[ik],ngrids], dtype=np.complex128)
        for ik in range(nkpts)]
    for ik in range(nkpts):
        Co_ks_G[ik] = tools.fft(Co_ks_R[ik].T, mesh) / ngrids**0.5

    return Co_ks_G


""" kinetic energy
"""
def apply_kin_kpt(mf, C_k, kpt, mesh=None, Gv=None):
    cell = mf.cell
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    no = C_k.shape[0]
    kG = kpt + Gv if np.sum(np.abs(kpt)) > 1.E-9 else Gv
    kG2 = np.einsum("gj,gj->g", kG, kG) * 0.5
    Cbar_k = C_k * kG2

    return Cbar_k


""" Pseudopotential
"""
def get_vpplocR(mf, Gv=None, mesh=None):
    cell = mf.cell
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    SI = cell.get_SI()
    vpplocG = pseudo.get_vlocG(cell, Gv)
    vpplocG = -np.einsum('ij,ij->j', SI, vpplocG)

    return tools.ifft(vpplocG, mesh).real


def apply_ppl_kpt(mf, C_k, kpt, mesh=None, Gv=None, vpplocR=None):

    cell = mf.cell
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)

    ngrids = Gv.shape[0]
    no = C_k.shape[0]
    fac = ngrids / cell.vol

    if vpplocR is None: vpplocR = mf.get_vpplocR(Gv=Gv, mesh=mesh)

    # local pp
    Cbar_k = tools.fft(tools.ifft(C_k, mesh) * vpplocR, mesh) * fac

    return Cbar_k


def apply_ppnl_kpt(mf, C_k, kpt, mesh=None, Gv=None):
    cell = mf.cell
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
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
                        tmp = hl @ np.einsum("imG,IG->Iim", SPG_lmi_, C_k_)
                        Cbar_k += np.einsum("Iim,imG->IG", tmp,
                            SPG_lmi_.conj())
        return Cbar_k / cell.vol

    Cbar_k = get_Cbar_k_nl(kpt, C_k)

    return Cbar_k


""" Columb energy
"""
def get_vj_R(mf, C_ks, mesh=None, Gv=None):
    cell = mf.cell
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    nkpts = len(C_ks)
    ngrids = np.prod(mesh)

    vj_G = np.zeros(ngrids, dtype=C_ks[0].dtype)
    for k in range(nkpts):
        C_k_R = tools.ifft(C_ks[k], mesh)
        vj_G += np.einsum("ig,ig->g", C_k_R.conj(), C_k_R)
    vj_G = tools.fft(vj_G, mesh)
    vj_G *= ngrids**2 / (cell.vol*nkpts)
    vj_G *= tools.get_coulG(cell, Gv=Gv)
    vj_R = tools.ifft(vj_G, mesh)

    return vj_R


def apply_vj_kpt(mf, C_k, kpt, mesh=None, Gv=None, vj_R=None):

    if vj_R is None: vj_R = mf.get_vj_R(C_ks, mesh=mesh, Gv=Gv)

    Cbar_k = tools.fft(tools.ifft(C_k, mesh) * vj_R, mesh)

    return Cbar_k


""" Exchange energy
"""
def apply_vk_kpt(mf, C_k, kpt1, C_ks, kpts, mesh=None, Gv=None, exxdiv=None):
    r""" Apply the EXX operator to given MOs

    Math:
        Cbar_k(G) = \sum_{j,k'} \sum_{G'} rho_{jk',ik}(G') v(k-k'+G') C_k(G-G')
    Code:
        rho_r = C_ik_r * C_jk'_r.conj()
        rho_G = FFT(rho_r)
        coulG = get_coulG(k-k')
        v_r = iFFT(rho_G * coulG)
        Cbar_ik_G = FFT(v_r * C_ik_r)
    """
    cell = mf.cell
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    ngrids = Gv.shape[0]
    nkpts = len(kpts)
    no_ks = [C_ks[jk].shape[0] for jk in range(nkpts)]
    fac = ngrids**2./(cell.vol*nkpts)

    mydf = df.FFTDF(cell)
    mydf.exxdiv = exxdiv

    Cbar_k = np.zeros_like(C_k)
    C_k_R = tools.ifft(C_k, mesh)

    for k2 in range(nkpts):
        kpt2 = kpts[k2]
        if exxdiv == 'ewald' or exxdiv is None:
            coulG = tools.get_coulG(cell, kpt1-kpt2, False, mydf, mesh)
        else:
            coulG = tools.get_coulG(cell, kpt1-kpt2, True, mydf, mesh)

        C_k2_R = tools.ifft(C_ks[k2], mesh)
        for j in range(no_ks[k2]):
            Cj_k2_R = C_k2_R[j]
            vij_R = tools.ifft(
                tools.fft(C_k_R * Cj_k2_R.conj(), mesh) * coulG, mesh)
            Cbar_k += vij_R * Cj_k2_R

    Cbar_k = tools.fft(Cbar_k, mesh) * fac

    return Cbar_k


def initialize_ACE(mf, C_ks, ace_exx=True, kpts=None, exxdiv=None):
    if not ace_exx:
        return None

    cell = mf.cell
    mesh = cell.mesh
    Gv = cell.get_Gv(mesh)
    if exxdiv is None: exxdiv = mf.exxdiv
    if kpts is None: kpts = mf.kpts
    nkpts = len(kpts)

    xi_ks = [None] * nkpts
    for k in range(nkpts):
        C_k = C_ks[k]
        W_k = mf.apply_vk_kpt(C_k, kpts[k], C_ks, kpts,
                              mesh=mesh, Gv=Gv, exxdiv=exxdiv)
        L_k = scipy.linalg.cholesky(C_k.conj()@W_k.T, lower=True)
        xi_ks[k] = scipy.linalg.solve_triangular(L_k.conj(), W_k, lower=True)

        # debug
        # W_k_prime = (C_k @ xi_ks[k].conj().T) @ xi_ks[k]
        # assert(np.linalg.norm(W_k - W_k_prime) < 1e-8)

    return xi_ks


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
    def __init__(self, mf, ndiis=5, diis_start=1):
        self.diis = DIIS()
        self.diis.space = ndiis
        self.diis.min_space = diis_start
        self.cycle = 0

    def next_step(self, mf, f, ferr):
        self.cycle += 1

        return self.diis.update(f, ferr)
