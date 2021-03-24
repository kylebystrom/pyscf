""" All actual implementation of PW-related PPs go here.
    The wrapper for calling the functions here go to pw_helper.py
"""

import tempfile
import numpy as np
import scipy.linalg
from scipy.special import dawsn, spherical_in

from pyscf.pbc.gto import pseudo as gth_pseudo
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import member
from pyscf import lib


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


def apply_vppl_kpt(cell, C_k, mesh=None, vpplocR=None):
    if mesh is None: mesh = cell.mesh
    if vpplocR is None: vpplocR = get_vpplocR(cell, mesh)
    return tools.fft(tools.ifft(C_k, mesh) * vpplocR, mesh)


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
        return apply_ppnl_kpt_alle(cell, C_k, kpt, Gv, **kwargs)


""" PW-PP class implementation goes here
"""
class PWPP:
    def __init__(self, cell, kpts, mesh=None):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.kpts = kpts
        if mesh is None: mesh = cell.mesh
        self.mesh = mesh
        self.Gv = cell.get_Gv(mesh)
        self.spin = None
        lib.logger.debug(self, "Initializing PP local part")
        self.vpplocR = get_vpplocR(cell, self.mesh, self.Gv)
        if len(cell._ecp) > 0:
            lib.logger.debug(self, "Initializing ccECP non-local part")
            self._ecp = format_ccecp_param(cell)
            self.fppnloc = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self.fswap = lib.H5TmpFile(self.fppnloc.name)
            nkpts = len(self.kpts)
            ngrids = self.Gv.shape[0]
            self.vppnlocGG = self.fswap.create_dataset(
                                        "vppnlocGG", shape=(nkpts,ngrids,ngrids),
                                        dtype=np.complex128)
            for k,kpt in enumerate(self.kpts):
                lib.logger.debug(self, "k = %d  kpt = [%.6f %.6f %.6f]",
                                 k, *kpt)
                self.vppnlocGG[k] = get_vppnlocGG_kpt_ccecp(cell, kpt, self.Gv,
                                                            _ecp=self._ecp)
        else:
            self.vppnlocGG = None

    def update_subspace_vppnloc(self, C_ks, spin=None):
        cell = self.cell
        if len(cell._ecp) > 0:
            dataname = "vppnlocWks"
            if dataname in self.fswap: del self.fswap[dataname]
            self.vppnlocWks = self.fswap.create_group(dataname)
            nkpts = len(self.kpts)
            for k in range(nkpts):
                key = "%d"%k if spin is None else "%d/%d" % (spin,k)
                C_k = C_ks[k] if isinstance(C_ks, list) else C_ks[key][()]
                W_k = lib.dot(C_k.conj(), self.vppnlocGG[k]).conj()
                M_k = lib.dot(C_k.conj(), W_k.T)
                e_k, u_k = scipy.linalg.eigh(M_k)
                idx_keep = np.where(e_k > 1e-12)[0]
                self.vppnlocWks[key] = lib.dot((u_k[:,idx_keep]*
                                                e_k[idx_keep]**-0.5).T, W_k)
                W_k = None
        else:
            self.vppnlocWks = None


    def apply_vppl_kpt(self, C_k, mesh=None, vpplocR=None):
        if mesh is None: mesh = self.mesh
        if vpplocR is None: vpplocR = self.vpplocR
        return apply_vppl_kpt(self, C_k, mesh=mesh, vpplocR=vpplocR)

    def apply_vppnl_kpt(self, C_k, kpt, mesh=None, Gv=None):
        cell = self.cell
        spin = self.spin
        if len(cell._ecp) > 0:
            k = member(kpt, self.kpts)[0]
            if self.vppnlocWks is None:
                return lib.dot(C_k.conj(), self.vppnlocGG[k]).conj()
            else:
                key = "%d"%k if spin is None else "%d/%d" % (spin,k)
                W_k = self.vppnlocWks[key][()]
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


def apply_ppnl_kpt_alle(cell, C_k, kpt, Gv):
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


def get_vppnlocGG_kpt_ccecp(cell, kpt, Gv, _ecp=None):
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
    buf = np.empty((48,ngrids), dtype=np.complex128)

    Gk = Gv + kpt
    G_rad = lib.norm(Gk, axis=1)
    G0idx = np.where(G_rad==0)[0]
    G_rad2 = G_rad[:,None] * G_rad
    with np.errstate(divide="ignore"):
        invG_rad = 1. / G_rad
        if G0idx.size > 0:
            invG_rad[G0idx] = 0

    vnlGG = np.zeros((ngrids,ngrids), dtype=np.complex128)
    for iatm in range(cell.natm):
        atm = cell.atom_symbol(iatm)
        _ecpnl_lst = _ecp[atm][1]
        for _ecpnl in _ecpnl_lst:
            l = _ecpnl[0]
            nl = (len(_ecpnl) - 1) // 2
            for il in range(nl):
                al, cl = _ecpnl[(1+il*2):(3+il*2)]
                fakemol._bas[0,gto.ANG_OF] = l
                fakemol._env[ptr+3] = 0.25 / al
                fakemol._env[ptr+4] = 2.*np.pi**1.25 * cl**0.5 / al**0.75
                # pYlm_part.shape = (ngrids, 2*l+1)
                pYlm_part = fakemol.eval_gto('GTOval', Gk) * SI[iatm][:,None]
                if l > 0:
                    pYlm_part *= (invG_rad**l)[:,None]
                vnlGG1 = lib.dot(pYlm_part, pYlm_part.conj().T)
                vnlGG2 = spherical_in(l, G_rad2 * (0.5/al))
                vnlGG += vnlGG1 * vnlGG2

    vnlGG /= cell.vol

    return vnlGG


def apply_vppnl_kpt_ccecp(cell, C_k, kpt, Gv, _ecp=None):
    """ very slow implementation
    """
    vppnlocGG = get_vppnlocGG_kpt_ccecp(cell, kpt, Gv, _ecp=_ecp)
    return lib.dot(C_k, vppnlocGG)
