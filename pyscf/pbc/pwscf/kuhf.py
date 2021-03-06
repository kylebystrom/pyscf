""" Spin-unrestricted Hartree-Fock in the Plane Wave Basis
"""


import time
import h5py
import copy
import numpy as np

from pyscf.pbc import gto, scf
from pyscf.pbc.pwscf import khf, pw_helper
from pyscf.lib import logger
from pyscf import __config__


def get_spin_component(C_ks, s):
    if isinstance(C_ks, list):
        return C_ks[s]
    else:
        return C_ks["%d"%s]


def remove_extra_virbands(mf, C_ks, moe_ks, mocc_ks, nbandv_extra):
    if isinstance(nbandv_extra, int): nbandv_extra = [nbandv_extra] * 2
    for s in [0,1]:
        C_ks_s = get_spin_component(C_ks, s)
        khf.remove_extra_virbands(mf, C_ks_s, moe_ks[s], mocc_ks[s],
                                   nbandv_extra[s])


def get_nband(mf, nbandv, nbandv_extra):
    cell = mf.cell
    if isinstance(nbandv, int): nbandv = [nbandv] * 2
    if isinstance(nbandv_extra, int): nbandv_extra = [nbandv_extra] * 2
    nbando = cell.nelec
    nbandv_tot = [nbandv[s] + nbandv_extra[s] for s in [0,1]]
    nband = [nbando[s] + nbandv[s] for s in [0,1]]
    nband_tot = [nbando[s] + nbandv_tot[s] for s in [0,1]]

    return nbando, nbandv_tot, nband, nband_tot


def get_band_err(mf, moe_ks, last_hf_moe, nband):
    return max([khf.get_band_err(mf, moe_ks[s], last_hf_moe[s], nband[s])
               for s in [0,1]])


def copy_C_ks(mf, C_ks, C_ks_exx):
    if C_ks_exx is None:
        return None
    else:
        for s in [0,1]:
            key = "%s" % s
            C_ks_s = C_ks[s] if isinstance(C_ks, list) else C_ks[key]
            if isinstance(C_ks_exx, list):
                if key in C_ks_exx: del C_ks_exx[key]
                C_ks_exx_s = C_ks_exx.create_group(key)
            else:
                C_ks_exx_s = C_ks_exx[s]
            khf.copy_C_ks(mf, C_ks_s, C_ks_exx_s)

        return C_ks_s


def dump_moe(mf, moe_ks, mocc_ks, nband=None, trigger_level=logger.DEBUG):
    if nband is None: nband = [None,None]
    if isinstance(nband, int): nband = [nband,nband]
    for s in [0,1]:
        khf.dump_moe(mf, moe_ks[s], mocc_ks[s],
                      nband=nband[s], trigger_level=trigger_level)


def get_mo_energy(mf, C_ks, mocc_ks, mesh=None, Gv=None, exxdiv=None,
                  C_ks_exx=None, ace_xi_ks=None, vpplocR=None, vj_R=None):

    moe_ks = [None] * 2
    for s in [0,1]:
        C_ks_ = get_spin_component(C_ks, s)
        C_ks_exx_ = None if C_ks_exx is None else \
                                            get_spin_component(C_ks_exx, s)
        ace_xi_ks_ = None if ace_xi_ks is None else \
                                            get_spin_component(ace_xi_ks, s)
        moe_ks[s] = khf.get_mo_energy(mf, C_ks_, mocc_ks[s],
                                       mesh=mesh, Gv=Gv, exxdiv=exxdiv,
                                       C_ks_exx=C_ks_exx_,
                                       ace_xi_ks=ace_xi_ks_,
                                       vpplocR=vpplocR, vj_R=vj_R,
                                       ret_mocc=False)

    # determine mo occ and apply ewald shift if requested
    mocc_ks = mf.get_mo_occ(moe_ks)
    if exxdiv is None: exxdiv = mf.exxdiv
    if exxdiv == "ewald":
        nkpts = len(mf.kpts)
        for s in [0,1]:
            for k in range(nkpts):
                moe_ks[s][k][mocc_ks[s][k] > khf.THR_OCC] -= mf._madelung

    return moe_ks, mocc_ks


def get_mo_occ(cell, moe_ks=None, C_ks=None):
    mocc_ks = [None] * 2
    for s in [0,1]:
        no = cell.nelec[s]
        if not moe_ks is None:
            mocc_ks[s] = khf.get_mo_occ(cell, moe_ks[s], no=no)
        elif not C_ks is None:
            C_ks_s = get_spin_component(C_ks, s)
            mocc_ks[s] = khf.get_mo_occ(cell, C_ks=C_ks_s, no=no)
        else:
            raise RuntimeError

    return mocc_ks


def get_init_guess(cell0, kpts, basis=None, pseudo=None, nv=0,
                   key="hcore", fC_ks=None):
    """
        Args:
            nv (int):
                Number of virtual bands to be evaluated. Default is zero.
            fC_ks (h5py group):
                If provided, the orbitals are written to it.
    """

    if not fC_ks is None:
        incore = False
        assert(isinstance(fC_ks, h5py.Group))
    else:
        incore = True

    nkpts = len(kpts)

    if basis is None: basis = cell0.basis
    if pseudo is None: pseudo = cell0.pseudo
    cell = copy.copy(cell0)
    cell.basis = basis
    cell.pseudo = pseudo
    cell.ke_cutoff = cell0.ke_cutoff
    cell.verbose = 0
    cell.build()

    logger.info(cell0, "generating init guess using %s basis", cell.basis)

    if len(kpts) < 30:
        pmf = scf.KUHF(cell, kpts)
    else:
        pmf = scf.KUHF(cell, kpts).density_fit()

    if key.lower() == "cycle1":
        pmf.max_cycle = 0
        pmf.kernel()
        mo_coeff = pmf.mo_coeff
        mo_occ = pmf.mo_occ
    elif key.lower() in ["hcore", "h1e"]:
        h1e = pmf.get_hcore()
        h1e = [h1e, h1e]
        s1e = pmf.get_ovlp()
        mo_energy, mo_coeff = pmf.eig(h1e, s1e)
        mo_occ = pmf.get_occ(mo_energy, mo_coeff)
    elif key.lower() == "scf":
        pmf.kernel()
        mo_coeff = pmf.mo_coeff
        mo_occ = pmf.mo_occ
    else:
        raise NotImplementedError("Init guess %s not implemented" % key)

    logger.debug1(cell0, "converting init MOs from GTO basis to PW basis")

    # TODO: support specifying nv for each kpt (useful for e.g., metals)
    if isinstance(nv, int): nv = [nv,nv]

    if incore: C_ks_spin = [None] * 2
    mocc_ks_spin = [None] * 2
    for s in [0,1]:
        no = cell.nelec[s]
        nmo_ks = [len(mo_occ[s][k]) for k in range(nkpts)]
        ntot = no + nv[s]
        ntot_ks = [min(ntot,nmo_ks[k]) for k in range(nkpts)]

        if incore:
            C_ks = pw_helper.get_C_ks_G(cell, kpts, mo_coeff[s], ntot_ks,
                                        verbose=cell0.verbose)
        else:
            key = "%d"%s
            if key in fC_ks: del fC_ks[key]
            C_ks = fC_ks.create_group(key)
            pw_helper.get_C_ks_G(cell, kpts, mo_coeff[s], ntot_ks,
                                 fC_ks=C_ks, verbose=cell0.verbose)
        mocc_ks = [mo_occ[s][k][:ntot_ks[k]] for k in range(nkpts)]

        C_ks = khf.orth_mo(cell0, C_ks, mocc_ks)

        C_ks, mocc_ks = khf.add_random_mo(cell0, [ntot]*nkpts, C_ks, mocc_ks)

        if incore: C_ks_spin[s] = C_ks
        mocc_ks_spin[s] = mocc_ks

    if not incore: C_ks_spin = fC_ks

    return C_ks_spin, mocc_ks_spin


def init_guess_by_chkfile(cell, chkfile_name, nv, project=None):
    fchk = h5py.File(chkfile_name, "a")
    C_ks = fchk["mo_coeff"]

    if isinstance(nv, int): nv = [nv] * 2

    from pyscf.pbc.scf import chkfile
    scf_dict = chkfile.load_scf(chkfile_name)[1]
    mocc_ks = scf_dict["mo_occ"]
    nkpts = len(mocc_ks[0])
    for s in [0,1]:
        ntot_ks = [None] * nkpts
        C_ks_s = C_ks["%d"%s]
        for k in range(nkpts):
            no = np.sum(mocc_ks[s][k]>khf.THR_OCC)
            ntot_ks[k] = max(no+nv[s], len(mocc_ks[s][k]))

        C_ks_s, mocc_ks[s] = khf.init_guess_from_C0(cell, C_ks_s, ntot_ks,
                                                    C_ks_s, mocc_ks[s])

    return fchk, C_ks, mocc_ks


def initialize_ACE(mf, C_ks, mocc_ks, kpts, mesh, Gv, ace_xi_ks, Ct_ks=None):
    tick = np.asarray([time.clock(), time.time()])
    if not "t-ace" in mf.scf_summary:
        mf.scf_summary["t-ace"] = np.zeros(2)

    if not ace_xi_ks is None:
        cell = mf.cell
        for s in [0,1]:
            C_ks_ = get_spin_component(C_ks, s)
            Ct_ks_ = None if Ct_ks is None else get_spin_component(Ct_ks, s)
            key = "%s" % s
            if key in ace_xi_ks: del ace_xi_ks[key]
            ace_xi_ks_spin = ace_xi_ks.create_group(key)
            pw_helper.initialize_ACE(cell, C_ks_, mocc_ks[s], kpts, mesh, Gv,
                                     ace_xi_ks_spin, Ct_ks=Ct_ks_)

    tock = np.asarray([time.clock(), time.time()])
    mf.scf_summary["t-ace"] += tock - tick


def energy_elec(mf, C_ks, mocc_ks, mesh=None, Gv=None, moe_ks=None,
                C_ks_exx=None, ace_xi_ks=None, vpplocR=None, vj_R=None,
                exxdiv=None):
    cell = mf.cell
    if vpplocR is None: vpplocR = mf.get_vpplocR()
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    if exxdiv is None: exxdiv = mf.exxdiv

    C_incore = isinstance(C_ks, list)

    kpts = mf.kpts
    nkpts = len(kpts)

    no_ks = [[np.sum(mocc_ks[s][k] > 0) for k in range(nkpts)] for s in [0,1]]

    e_ks = np.zeros(nkpts)
    if moe_ks is None:
        if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)
        e_comp = np.zeros(5)
        for s in [0,1]:
            C_ks_s = get_spin_component(C_ks, s)
            C_ks_exx_s = None if C_ks_exx is None else \
                                                get_spin_component(C_ks_exx, s)
            ace_xi_ks_s = None if ace_xi_ks is None else \
                                                get_spin_component(ace_xi_ks, s)
            for k in range(nkpts):
                kpt = kpts[k]
                no_k = no_ks[s][k]
                Co_k = C_ks_s[k][:no_k] if C_incore else C_ks_s["%d"%k][:no_k]
                ace_xi_k = None if ace_xi_ks_s is None else \
                                                        ace_xi_ks_s["%d"%k][()]
                e_comp_k = mf.apply_Fock_kpt(Co_k, kpt, mocc_ks, mesh, Gv,
                                             vpplocR, vj_R, exxdiv,
                                             C_ks_exx=C_ks_exx_s, ace_xi_k=ace_xi_k,
                                             ret_E=True)[1]
                e_comp_k *= 0.5
                e_ks[k] += np.sum(e_comp_k)
                e_comp += e_comp_k
        e_comp /= nkpts

        if exxdiv == "ewald":
            e_comp[mf.scf_summary["e_comp_name_lst"].index("ex")] += \
                                                        mf._etot_shift_ewald

        for comp,e in zip(mf.scf_summary["e_comp_name_lst"],e_comp):
            mf.scf_summary[comp] = e
    else:
        for s in [0,1]:
            C_ks_s = get_spin_component(C_ks, s)
            moe_ks_s = moe_ks[s]
            for k in range(nkpts):
                kpt = kpts[k]
                no_k = no_ks[s][k]
                Co_k = C_ks_s[k][:no_k] if C_incore else C_ks_s["%d"%k][:no_k]
                e1_comp = mf.apply_h1e_kpt(Co_k, kpt, mesh, Gv,
                                           vpplocR, ret_E=True)[1]
                e1_comp *= 0.5
                e_ks[k] += np.sum(e1_comp) * 0.5 + np.sum(moe_ks_s[k][:no_k]) * 0.5
    e_scf = np.sum(e_ks) / nkpts

    if moe_ks is None and exxdiv == "ewald":
        # Note: ewald correction is not needed if e_tot is computed from moe_ks since the correction is already in the mo energy
        e_scf += mf._etot_shift_ewald

    return e_scf


def converge_band(mf, C_ks, mocc_ks, kpts, Cout_ks=None,
                  mesh=None, Gv=None,
                  C_ks_exx=None, ace_xi_ks=None,
                  vpplocR=None, vj_R=None,
                  conv_tol_davidson=1e-6,
                  max_cycle_davidson=100,
                  verbose_davidson=0):

    nkpts = len(kpts)

    conv_ks = [None] * 2
    moeout_ks = [None] * 2
    fc_ks = [None] * 2
    if isinstance(C_ks, list):
        if Cout_ks is None: Cout_ks = [None] * 2
    else:
        Cout_ks = C_ks
    for s in [0,1]:
        C_ks_s = get_spin_component(C_ks, s)
        C_ks_exx_s = None if C_ks_exx is None else \
                                            get_spin_component(C_ks_exx, s)
        ace_xi_ks_s = None if ace_xi_ks is None else \
                                            get_spin_component(ace_xi_ks, s)
        conv_ks[s], moeout_ks[s], Cout_ks_s, fc_ks[s] = khf.converge_band(
                            mf, C_ks_s, mocc_ks[s], kpts, mesh=mesh, Gv=Gv,
                            C_ks_exx=C_ks_exx_s, ace_xi_ks=ace_xi_ks_s,
                            vpplocR=vpplocR, vj_R=vj_R,
                            conv_tol_davidson=conv_tol_davidson,
                            max_cycle_davidson=max_cycle_davidson,
                            verbose_davidson=verbose_davidson)

        if isinstance(C_ks, list): Cout_ks[s] = Cout_ks_s

    fc_ks = [fc_ks[0][k]+fc_ks[1][k] for k in range(nkpts)]

    return conv_ks, moeout_ks, Cout_ks, fc_ks


class PWKUHF(khf.PWKRHF):

    def __init__(self, cell, kpts=np.zeros((1,3)), ekincut=None,
        exxdiv=getattr(__config__, 'pbc_scf_PWKUHF_exxdiv', 'ewald')):

        khf.PWKRHF.__init__(self, cell, kpts=kpts, exxdiv=exxdiv)

        self.nv = [0,0]
        self.nv_extra = [1,1]

    def get_init_guess_key(self, cell=None, kpts=None, basis=None, pseudo=None,
                           nv=None, key="hcore", fC_ks=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if nv is None: nv = self.nv

        if key in ["h1e","hcore","cycle1","scf"]:
            C_ks, mocc_ks = get_init_guess(cell, kpts,
                                           basis=basis, pseudo=pseudo,
                                           nv=nv, key=key, fC_ks=fC_ks)
        else:
            logger.warn(self, "Unknown init guess %s", key)
            raise RuntimeError

        return C_ks, mocc_ks

    def get_init_guess_C0(self, C0, nv=None, fC_ks=None):
        if nv is None: nv = self.nv
        if isinstance(nv, int): nv = [nv,nv]
        no = self.cell.nelec
        nkpts = len(self.kpts)
        incore0 = isinstance(C0, list)
        incore = fC_ks is None
        C_ks = [None] * 2 if incore else fC_ks
        mocc_ks = [None] * 2
        for s in [0,1]:
            ntot_ks = [no[s]+nv[s]] * len(self.kpts)
            if incore:
                C_ks_s = None
            else:
                key = "%d" % s
                if key in C_ks: del C_ks[key]
                C_ks_s = C_ks.create_group(key)
            if incore0:
                C0_s = C0[s]
                n0_ks = [C0_s[k].shape[0] for k in range(nkpts)]
            else:
                C0_s = C0["%d"%s]
                n0_ks = [C0_s["%d"%k].shape[0] for k in range(nkpts)]
            mocc_ks[s] = [np.asarray([1 if i < no[s] else 0
                          for i in range(n0_ks[k])]) for k in range(nkpts)]
            C_ks_s, mocc_ks[s] = khf.init_guess_from_C0(self.cell, C0_s,
                                                        ntot_ks, C_ks_s,
                                                        mocc_ks[s])
            if incore: C_ks[s] = C_ks_s

        return C_ks, mocc_ks

    def init_guess_by_chkfile(self, chk=None, nv=None, project=None):
        if chk is None: chk = self.chkfile
        if nv is None: nv = self.nv
        # return init_guess_by_chkfile(self.cell, chk, project)
        return init_guess_by_chkfile(self.cell, chk, nv, project=project)
    def from_chk(self, chk=None, project=None, kpts=None):
        return self.init_guess_by_chkfile(chk, project, kpts)

    def get_mo_occ(mf, moe_ks=None, C_ks=None):
        return get_mo_occ(mf.cell, moe_ks, C_ks)

    def get_vj_R(self, C_ks, mocc_ks, mesh=None, Gv=None):
        vj_R = 0. + 0.j
        for s in [0,1]:
            C_ks_ = get_spin_component(C_ks, s)
            vj_R += pw_helper.get_vj_R(self.cell, C_ks_, mocc_ks[s],
                                       mesh=mesh, Gv=Gv)
        vj_R *= 0.5
        return vj_R

    remove_extra_virbands = remove_extra_virbands
    get_nband = get_nband
    get_band_err = get_band_err
    dump_moe = dump_moe
    copy_C_ks = copy_C_ks
    initialize_ACE = initialize_ACE
    get_mo_energy = get_mo_energy
    energy_elec = energy_elec
    converge_band = converge_band


if __name__ == "__main__":
    cell = gto.Cell(
        atom = "C 0 0 0",
        a = np.eye(3) * 6,
        basis="gth-szv",
        ke_cutoff=50,
        pseudo="gth-pade",
        spin=2,
    )
    cell.build()
    cell.verbose = 5

    nk = 1
    kmesh = (nk,)*3
    kpts = cell.make_kpts(kmesh)

    umf = PWKUHF(cell, kpts)
    umf.nv = [0,2]
    umf.chkfile = "mf.chk"
    umf.init_guess = "chk"
    umf.kernel()

    umf.dump_scf_summary()

    assert(abs(umf.e_tot - -5.365678833) < 1e-5)
