""" All actual implementation of PW-related PPs go here.
    The wrapper for calling the functions here go to pw_helper.py
"""

import tempfile
import numpy as np
import scipy.linalg
from scipy.special import dawsn

from pyscf.pbc.pwscf.pw_helper import get_kcomp, set_kcomp
from pyscf.pbc.gto import pseudo as gth_pseudo
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import member
from pyscf import lib
from pyscf import __config__


""" Wrapper functions
"""
def get_vpplocR(cell, mesh=None, Gv=None):
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh=mesh)
    SI = cell.get_SI(Gv=Gv)
    ngrids = Gv.shape[0]
    fac = ngrids / cell.vol
    vpplocG = np.einsum("ag,ag->g", SI, get_vpplocG(cell, mesh, Gv))
    vpplocR = tools.ifft(vpplocG, mesh).real * fac

    return vpplocR


def get_vpplocG(cell, mesh=None, Gv=None):
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh=mesh)

    if len(cell._ecp) > 0:
        return get_vpplocG_ccecp(cell, Gv)
    elif not cell.pseudo is None:
        if "GTH" in cell.pseudo.upper():
            return get_vpplocG_gth(cell, Gv)
        else:
            raise NotImplementedError("Pseudopotential %s is currently not supported." % (str(cell.pseudo)))
    else:
        return get_vpplocG_alle(cell, Gv)


def apply_vppl_kpt(cell, C_k, mesh=None, vpplocR=None, C_k_R=None):
    if mesh is None: mesh = cell.mesh
    if vpplocR is None: vpplocR = get_vpplocR(cell, mesh)
    if C_k_R is None: C_k_R = tools.ifft(C_k, mesh)
    return tools.fft(C_k_R * vpplocR, mesh)


def apply_vppnl_kpt(cell, C_k, kpt, mesh=None, Gv=None):
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh=mesh)

    if len(cell._ecp) > 0:
        return apply_vppnl_kpt_ccecp(cell, C_k, kpt, Gv)
    elif not cell.pseudo is None:
        if "GTH" in cell.pseudo.upper():
            return apply_vppnl_kpt_gth(cell, C_k, kpt, Gv)
        else:
            raise NotImplementedError("Pseudopotential %s is currently not supported." % (str(cell.pseudo)))
    else:
        return apply_vppnl_kpt_alle(cell, C_k, kpt, Gv, **kwargs)


""" PW-PP class implementation goes here
"""
def pseudopotential(mf, with_pp=None, mesh=None):
    if with_pp is None:
        with_pp = PWPP(mf.cell, mf.kpts, mesh=mesh)

    mf.with_pp = with_pp

    return mf


class PWPP:

    direct_nloc = getattr(__config__, "pbc_pwscf_pseudo_PWPP_direct_nloc", True)

    def __init__(self, cell, kpts, mesh=None):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.kpts = kpts
        if mesh is None: mesh = cell.mesh
        self.mesh = mesh
        self.Gv = cell.get_Gv(mesh)
        lib.logger.debug(self, "Initializing PP local part")
        self.vpplocR = get_vpplocR(cell, self.mesh, self.Gv)
        if len(cell._ecp) > 0:
            lib.logger.debug(self, "Initializing ccECP non-local part")
            dtype = np.complex128
            self._ecp = format_ccecp_param(cell)
            self.swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self.fswap = lib.H5TmpFile(self.swapfile.name)
            if self.direct_nloc:
                self.vppnlocGG = None
                self.vppnlocWks = self.fswap.create_group("vppnlocWks")
            else:
                nkpts = len(self.kpts)
                ngrids = self.Gv.shape[0]
                self.vppnlocGG = self.fswap.create_dataset(
                                                    "vppnlocGG", shape=(nkpts,ngrids,ngrids),
                                                    dtype=dtype)
                fill_vppnlocGG_ccecp(cell, self.kpts, self.Gv, _ecp=self._ecp,
                                     out=self.vppnlocGG)
        else:
            self._ecp = None
            self.vppnlocGG = None

    def update_vppnloc_support_vec(self, C_ks, ncomp=1):
        if not self._ecp is None:
            out = self.vppnlocWks
            if ncomp > 1:
                for comp in range(ncomp):
                    key = "%d"%comp
                    if key in out: del out[key]
                    out.create_group(key)
            nkpts = len(self.kpts)
            cell = self.cell
            for k in range(nkpts):
                if ncomp == 1:
                    C_k = get_kcomp(C_ks, k)
                else:
                    comp_loc = [0] * (ncomp+1)
                    C_k = [None] * ncomp
                    for comp in range(ncomp):
                        C_k[comp] = get_kcomp(C_ks["%d"%comp], k)
                        comp_loc[comp+1] = comp_loc[comp] + C_k[comp].shape[0]
                    C_k = np.vstack(C_k)
                if self.vppnlocGG is None:
                    kpt = self.kpts[k]
                    Gv = self.Gv
                    W_k = apply_vppnlocGG_kpt_ccecp(cell, C_k, kpt, Gv,
                                                    _ecp=self._ecp)
                else:
                    W_k = apply_vppnlocGG_kpt_ccecp_precompute(cell, C_k, k,
                                                               self.vppnlocGG)
                if ncomp == 1:
                    W_k = get_support_vec(C_k, W_k, method="eig")
                    set_kcomp(W_k, out, k)
                else:
                    for comp in range(ncomp):
                        p0, p1 = comp_loc[comp:comp+2]
                        w_k = get_support_vec(C_k[p0:p1], W_k[p0:p1],
                                              method="eig")
                        set_kcomp(w_k, out["%d"%comp], k)
                        w_k = None

                C_k = W_k = None

    def apply_vppl_kpt(self, C_k, mesh=None, vpplocR=None, C_k_R=None):
        if mesh is None: mesh = self.mesh
        if vpplocR is None: vpplocR = self.vpplocR
        return apply_vppl_kpt(self, C_k, mesh=mesh, vpplocR=vpplocR,
                              C_k_R=C_k_R)

    def apply_vppnl_kpt(self, C_k, kpt, mesh=None, Gv=None, comp=None):
        cell = self.cell
        if len(cell._ecp) > 0:
            k = member(kpt, self.kpts)[0]
            if self.vppnlocWks is None:
                return lib.dot(C_k.conj(), self.vppnlocGG[k]).conj()
            else:
                if comp is None:
                    W_k = get_kcomp(self.vppnlocWks, k)
                elif isinstance(comp, int):
                    W_k = get_kcomp(self.vppnlocWks["%d"%comp], k)
                else:
                    raise RuntimeError("comp must be None or int")
                return lib.dot(lib.dot(C_k, W_k.T.conj()), W_k)
        elif not cell.pseudo is None:
            if "GTH" in cell.pseudo.upper():
                return apply_vppnl_kpt_gth(cell, C_k, kpt, Gv)
            else:
                raise NotImplementedError("Pseudopotential %s is currently not supported." % (str(cell.pseudo)))
        else:
            return apply_vppnl_kpt_alle(cell, C_k, kpt, Gv)


""" All-electron implementation starts here
"""
def get_vpplocG_alle(cell, Gv):
    Zs = cell.atom_charges()
    coulG = tools.get_coulG(cell, Gv=Gv)
    vpplocG = -np.einsum("a,g->ag", Zs, coulG)
    return vpplocG


def apply_vppnl_kpt_alle(cell, C_k, kpt, Gv):
    return np.zeros_like(C_k)


""" GTH implementation starts here
"""
def get_vpplocG_gth(cell, Gv):
    return -gth_pseudo.get_vlocG(cell, Gv)


def apply_vppnl_kpt_gth(cell, C_k, kpt, Gv):
    SI = cell.get_SI(Gv=Gv)
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
                        qkl = gth_pseudo.pp._qli(G_rad*rl, l, k)
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


""" ccECP implementation starts here
"""
def fast_SphBslin(n, xs, thr_switch=20, thr_overflow=700, out=None):
    if out is None: out = np.zeros_like(xs)
    if n == 0:
        out[:] = np.sinh(xs) / xs
    elif n == 1:
        out[:] = (xs * np.cosh(xs) - np.sinh(xs)) / xs**2.
    elif n == 2:
        out[:] = ((xs**2.+3.)*np.sinh(xs) - 3.*xs*np.cosh(xs)) / xs**3.
    elif n == 3:
        out[:] = ((xs**3.+15.*xs)*np.cosh(xs) -
                (6.*xs**2.+15.)*np.sinh(xs)) / xs**4.
    else:
        raise NotImplementedError("fast_SphBslin with n=%d is not implemented." % n)

    return out


def format_ccecp_param(cell):
    r""" Format the ecp data into the following dictionary:
        _ecp = {
                    atm1: [_ecpl_atm1, _ecpnl_atm1],
                    atm2: [_ecpl_atm2, _ecpnl_atm2],
                    ...
                }
        _ecpl  = [alp_1, c_1=Zeff, alp_2, c_2, alp1_3, c1_3, alp2_3, c2_3, ...]
        _ecpnl = [
                    [l1, alp1_l1, c1_l1, alp2_l1, c2_l1, ...],
                    [l2, alp1_l2, c1_l2, alp2_l2, c2_l2, ...],
                    ...
                ]
        where
            Vl(r)  = -Zeff/r + c_1/r*exp(-alp_1*r^2) + c_2*r*exp(-alp_2*r^2) +
                        \sum_{k} ck_3*exp(-alpk_3*r^2)
            Vnl(r) = \sum_l \sum_k ck_l * exp(-alpk_l*r^2) \sum_m |lm><lm|
    """
    uniq_atms = cell._basis.keys()
    _ecp = {}
    for iatm in range(cell.natm):
        atm = cell.atom_symbol(iatm)
        if atm in _ecp: continue
        ncore, ecp_dic = cell._ecp[atm]
# local part
        ecp_loc = ecp_dic[0]
        _ecp_loc = []
        ecp_loc_item = ecp_loc[1]
        _ecp_loc = np.concatenate([ecp_loc_item[1][0], ecp_loc_item[3][0],
                                  *ecp_loc_item[2]])
# non-local part
        _ecp_nloc = []
        for ecp_nloc_litem in ecp_dic[1:]:
            l = ecp_nloc_litem[0]
            _ecp_nloc_item = [l]
            for ecp_nloc_item in ecp_nloc_litem[1]:
                if len(ecp_nloc_item) > 0:
                    for ecp_nloc_item2 in ecp_nloc_item:
                        _ecp_nloc_item += ecp_nloc_item2
            _ecp_nloc.append(_ecp_nloc_item)
        _ecp[atm] = [_ecp_loc, _ecp_nloc]

    return _ecp


def get_vpplocG_ccecp(cell, Gv, _ecp=None):
    if _ecp is None: _ecp = format_ccecp_param(cell)
    G_rad = np.linalg.norm(Gv, axis=1)
    coulG = tools.get_coulG(cell, Gv=Gv)
    G0_idx = np.where(G_rad==0)[0]
    with np.errstate(divide="ignore"):
        invG = 4*np.pi / G_rad
        invG[G0_idx] = 0
    ngrids = coulG.size
    vlocG = np.zeros((cell.natm,ngrids))
    for iatm in range(cell.natm):
        atm = cell.atom_symbol(iatm)
        _ecpi = _ecp[atm][0]
# Zeff / r
        Zeff = _ecpi[1]
        vlocG[iatm] += -coulG * Zeff
        v0 = -coulG[G0_idx] * Zeff
# c1 / r * exp(-a1 * r^2)
        a1, c1 = _ecpi[:2]
        vlocG[iatm] += c1 * invG * a1**-0.5 * dawsn(G_rad*(0.5/a1**0.5))
        v0 += 2*np.pi / a1 * c1
# c2 * r * exp(-a2 * r^2)
        a2, c2 = _ecpi[2:4]
        vlocG[iatm] += c2 * (np.pi/a2**2. + ((0.5/a2**1.5) * invG -
                                             (np.pi/a2**2.5)*G_rad) *
                             dawsn(G_rad*(0.5/a2**0.5)))
        v0 += 2*np.pi / a2**2 * c2
# \sum_k c3_k * exp(-a3_k * r^2)
        n3 = (len(_ecpi) - 4) // 2
        if n3 > 0:
            for i3 in range(n3):
                a3, c3 = _ecpi[(4+i3*2):(6+i3*2)]
                vlocG[iatm] += c3 * (np.pi/a3)**1.5 * np.exp(-G_rad**2.*
                                                             (0.25/a3))
                v0 += (np.pi/a3)**1.5 * c3
# G = 0
        vlocG[iatm][G0_idx] = v0

    return vlocG


def fill_vppnlocGG_ccecp(cell, kpts, Gv, out, _ecp=None):
    if _ecp is None: _ecp = format_ccecp_param(cell)
    SI = cell.get_SI()
    ngrids = Gv.shape[0]

    dsize = 16
    max_memory = (cell.max_memory - lib.current_memory()[0]) * 0.4
    Gblksize = min(int(np.floor(max_memory*1e6/dsize / ngrids)), ngrids)
    lib.logger.debug1(cell, "fill in vppnlocGG_ccecp in %d segs",
                      (ngrids-1)//Gblksize+1)

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

# get uniq atom list
    uniq_atm_map = dict()
    for iatm in range(cell.natm):
        atm = cell.atom_symbol(iatm)
        if not atm in uniq_atm_map:
            uniq_atm_map[atm] = []
        uniq_atm_map[atm].append(iatm)

    for k,kpt in enumerate(kpts):
        for p0,p1 in lib.prange(0,ngrids,Gblksize):
            out[k,p0:p1] = 0. + 0.j

        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)
        if abs(kpt).sum() < 1e-8: G_rad += 1e-40    # avoid inverting zero
        G_rad2 = G_rad[:,None] * G_rad
        invG_rad = 1. / G_rad

        for atm,iatm_lst in uniq_atm_map.items():
            _ecpnl_lst = _ecp[atm][1]
            for _ecpnl in _ecpnl_lst:
                l = _ecpnl[0]
                nl = (len(_ecpnl) - 1) // 2
                for il in range(nl):
                    al, cl = _ecpnl[(1+il*2):(3+il*2)]
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = 0.25 / al
                    fakemol._env[ptr+4] = 2.*np.pi**1.25 * cl**0.5 / al**0.75
                    # pYlm_part.shape = (ngrids, (2*l+1)*len(iatm_lst))
                    pYlm_part = np.einsum("gl,ag->gla",
                                          fakemol.eval_gto('GTOval', Gk),
                                          SI[iatm_lst]).reshape(ngrids,-1)
                    if l > 0:
                        pYlm_part *= (invG_rad**l)[:,None]
                    for p0,p1 in lib.prange(0,ngrids,Gblksize):
                        vnlGG = lib.dot(pYlm_part[p0:p1], pYlm_part.conj().T)
                        vnlGG *= fast_SphBslin(l, G_rad2[p0:p1] * (0.5/al))
                        out[k,p0:p1] += vnlGG
                        vnlGG = None

        for p0,p1 in lib.prange(0,ngrids,Gblksize):
            out[k,p0:p1] /= cell.vol

    return vnlGG


def apply_vppnlocGG_kpt_ccecp(cell, C_k, kpt, Gv, _ecp=None):
    if _ecp is None: _ecp = format_ccecp_param(cell)
    SI = cell.get_SI()
    ngrids = Gv.shape[0]

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

    uniq_atm_map = dict()
    for iatm in range(cell.natm):
        atm = cell.atom_symbol(iatm)
        if not atm in uniq_atm_map:
            uniq_atm_map[atm] = []
        uniq_atm_map[atm].append(iatm)

    nmo = C_k.shape[0]
    lmax = np.max([_ecpitem[1][0] for _ecpitem in _ecp.values()])
    natmmax = np.max([len(iatm_lst) for iatm_lst in uniq_atm_map.values()])

    dtype = np.complex128
    dsize = 16
    max_memory = (cell.max_memory - lib.current_memory()[0]) * 0.9
    Gblksize = min(int(np.floor((max_memory*1e6/dsize/ngrids -
                                 ((2*lmax+1)*natmmax+3+nmo))*0.3)), ngrids)
    buf = np.empty(Gblksize*ngrids, dtype=dtype)
    buf2 = np.empty(Gblksize*ngrids, dtype=dtype)
    buf3 = np.empty(Gblksize*ngrids, dtype=dtype)
    lib.logger.debug1(cell, "Computing v^nl*C_k in %d segs with blksize %d",
                      (ngrids-1)//Gblksize+1, Gblksize)

    Gk = Gv + kpt
    G_rad = lib.norm(Gk, axis=1)
    if abs(kpt).sum() < 1e-8: G_rad += 1e-40    # avoid inverting zero

    Cbar_k = np.zeros_like(C_k)
    for atm,iatm_lst in uniq_atm_map.items():
        _ecpnl_lst = _ecp[atm][1]
        for _ecpnl in _ecpnl_lst:
            l = _ecpnl[0]
            nl = (len(_ecpnl) - 1) // 2
            for il in range(nl):
                al, cl = _ecpnl[(1+il*2):(3+il*2)]
                fakemol._bas[0,gto.ANG_OF] = l
                fakemol._env[ptr+3] = 0.25 / al
                fakemol._env[ptr+4] = 2.*np.pi**1.25 * cl**0.5 / al**0.75
                # pYlm_part.shape = (ngrids, (2*l+1)*len(iatm_lst))
                pYlm_part = np.einsum("gl,ag->gla",
                                      fakemol.eval_gto('GTOval', Gk),
                                      SI[iatm_lst]).reshape(ngrids,-1)
                if l > 0:
                    pYlm_part[:] *= (invG_rad**l)[:,None]
                G_red = G_rad * (0.5 / al)
                for p0,p1 in lib.prange(0,ngrids,Gblksize):
                    G_rad2 = np.ndarray((p1-p0,ngrids), dtype=dtype, buffer=buf)
                    vnlGG = np.ndarray((p1-p0,ngrids), dtype=dtype, buffer=buf2)
                    SBin = np.ndarray((p1-p0,ngrids), dtype=dtype, buffer=buf3)
                    G_rad2[:] = G_rad[p0:p1,None]*G_red
                    # use np.dot since a slice is neither F nor C-contiguous
                    vnlGG = np.dot(pYlm_part[p0:p1], pYlm_part.conj().T,
                                   out=vnlGG)
                    vnlGG[:] *= fast_SphBslin(l, G_rad2, out=SBin)
                    Cbar_k[:,p0:p1] += lib.dot(vnlGG, C_k.T).T
                    G_rad2 = vnlGG = SBin = None
                G_red = pYlm_part = None
    Cbar_k /= cell.vol

    return Cbar_k


def apply_vppnlocGG_kpt_ccecp_precompute(cell, C_k, k, vppnlocGG):
    ngrids = C_k.shape[1]
    max_memory = (cell.max_memory - lib.current_memory()[0]) * 0.8
    Gblksize = min(int(np.floor(max_memory*1e6/16/ngrids)), ngrids)
    W_k = np.zeros_like(C_k)
    for p0,p1 in lib.prange(0,ngrids,Gblksize):
        W_k += lib.dot(C_k[:,p0:p1].conj(), vppnlocGG[k,p0:p1]).conj()
    return W_k


def apply_vppnl_kpt_ccecp(cell, C_k, kpt, Gv, _ecp=None):
    """ very slow implementation
    """
    vppnlocGG = get_vppnlocGG_kpt_ccecp(cell, kpt, Gv, _ecp=_ecp)
    return lib.dot(C_k, vppnlocGG)


def get_support_vec(C, W, method="cd", thr_eig=1e-12):
    M = lib.dot(C.conj(), W.T)
    if np.sum(np.abs(M)) < 1e-10:
        svec = np.zeros_like(C)
    else:
        if method == "cd":
            svec = scipy.linalg.cholesky(M, lower=True)
            svec = scipy.linalg.solve_triangular(svec.conj(), W, lower=True)
        elif method == "eig":
            e, u = scipy.linalg.eigh(M)
            idx_keep = np.where(e > thr_eig)[0]
            svec = lib.dot((u[:,idx_keep]*e[idx_keep]**-0.5).T, W)
        else:
            raise RuntimeError("Unknown method %s" % str(method))

    return svec
