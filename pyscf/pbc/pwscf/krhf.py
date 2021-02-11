""" Hartree-Fock in the Plane Wave Basis
"""


import time
import copy
import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf import __config__
from pyscf.scf import hf as mol_hf
from pyscf.scf import chkfile
from pyscf.pbc import gto, scf, tools
from pyscf.pbc.pwscf import pw_helper
from pyscf.lib import logger
import pyscf.lib.parameters as param


def kernel(mf, kpts, C0_ks=None, conv_tol=1.E-6, conv_tol_davidson=1.E-6,
           max_cycle=100, max_cycle_davidson=100, verbose_davidson=0,
           dump_chk=True, conv_check=True, callback=None, **kwargs):
    ''' Kernel function for SCF in a PW basis
    '''

    cput0 = (time.clock(), time.time())

    cell = mf.cell
    nkpts = len(kpts)

    if C0_ks is None:
        C0_ks = mf.get_init_guess()

    # init E
    vpplocR = mf.get_vpplocR()
    vj_R = mf.get_vj_R(C0_ks)
    moe_ks = mf.get_mo_energy(C0_ks, vpplocR=vpplocR, vj_R=vj_R)
    e_tot = mf.energy_tot(C0_ks, moe_ks=moe_ks, vpplocR=vpplocR)
    logger.info(mf, 'init E= %.15g', e_tot)
    mf.dump_moe(moe_ks)

    scf_conv = False

    if mf.max_cycle <= 0:
        return scf_conv, e_tot, moe_ks, C0_ks

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(cell, mf.chkfile)

    fc_tot = 0
    fc_this = 0
    cput1 = logger.timer(mf, 'initialize pwscf', *cput0)
    for cycle in range(max_cycle):

        if cycle > 0:   # already built for init E at cycle 0
            vpplocR = mf.get_vpplocR()
            vj_R = mf.get_vj_R(C0_ks)

        conv_ks, moe_ks, C0_ks, fc_ks = mf.converge_band(
                            C0_ks, kpts, moe_ks=moe_ks,
                            vpplocR=vpplocR, vj_R=vj_R,
                            conv_tol_davidson=conv_tol_davidson,
                            max_cycle_davidson=max_cycle_davidson,
                            verbose_davidson=verbose_davidson)
        fc_this = sum(fc_ks)
        fc_tot += fc_this

        last_hf_e = e_tot
        e_tot = mf.energy_tot(C0_ks, moe_ks=moe_ks, vpplocR=vpplocR)
        de = e_tot-last_hf_e if cycle > 0 else 10
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  %d FC (%d tot)',
                    cycle+1, e_tot, de, fc_this, fc_tot)
        mf.dump_moe(moe_ks)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break

    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        conv_ks, moe_ks, C0_ks, fc_ks = mf.converge_band(
                            C0_ks, kpts, moe_ks=moe_ks,
                            vpplocR=vpplocR, vj_R=vj_R,
                            conv_tol_davidson=conv_tol_davidson,
                            max_cycle_davidson=max_cycle_davidson,
                            verbose_davidson=verbose_davidson)
        fc_this = sum(fc_ks)
        fc_tot += fc_this
        last_hf_e = e_tot
        e_tot = mf.energy_tot(C0_ks, moe_ks=moe_ks, vpplocR=vpplocR)
        de = e_tot-last_hf_e if cycle > 0 else 10
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol:
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  %d FC (%d tot)',
                    e_tot, de, fc_this, fc_tot)
        mf.dump_moe(moe_ks)

        if dump_chk:
            mf.dump_chk(locals())

    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel(locals())
    return scf_conv, e_tot, moe_ks, C0_ks


def dump_moe(mf, moe_ks):
    if mf.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=len(moe_ks))
        logger.debug(mf, '     k-point                  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                         k, kpt[0], kpt[1], kpt[2], moe_ks[k].real)
        np.set_printoptions(threshold=1000)


def get_init_guess(mf):
    cell = mf.cell
    kpts = mf.kpts

    logger.info(mf, "generating init guess using %s basis", cell.basis)

    verbose = cell.verbose
    cell.verbose = 0
    if len(kpts) < 30:
        pmf = scf.KRHF(cell, kpts)
    else:
        pmf = scf.KRHF(cell, kpts).density_fit()
    cell.verbose = verbose
    pmf.exxdiv = mf.exxdiv
    pmf.max_cycle = 0
    pmf.kernel()

    cell.ke_cutoff = cell.ke_cutoff
    cell.build()
    Co_ks = pw_helper.get_Co_ks_G(cell, kpts, pmf.mo_coeff, pmf.mo_occ)
    for i,kpt in enumerate(kpts):
        Sk = Co_ks[i].conj() @ Co_ks[i].T
        nonorth_err = np.max(np.abs(Sk - np.eye(Sk.shape[0])))
        if nonorth_err > mf.conv_tol * 1e-3:
            logger.warn(mf, "non-orthogonality detected in the initial MOs (max |off-diag ovlp|= %s) for kpt %d. Symm-orth them now.", nonorth_err, i)
        e, u = scipy.linalg.eigh(Sk)
        Co_ks[i] = (u*e**-0.5).T @ Co_ks[i]

    return Co_ks


def apply_h1e_kpt(mf, C_k, kpt, cell=None, vpplocR=None, ret_E=False):
    r''' Apply 1e part of the Fock operator to orbitals at given k-points.
        Math:
            |psibar_ik> = (hat{T} + hat{vpp}) |psi_ik>, for all i and k = kpt
    '''
    if cell is None: cell = mf.cell
    if vpplocR is None: vpplocR = mf.get_vpplocR()

    mesh = cell.mesh
    Gv = cell.get_Gv(mesh)

    es = np.zeros([3], dtype=np.complex128)

    tmp = mf.apply_kin_kpt(C_k, kpt, mesh=mesh, Gv=Gv)
    Cbar_k = tmp
    es[0] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2

    tmp = mf.apply_ppl_kpt(C_k, kpt, mesh=mesh, Gv=Gv, vpplocR=vpplocR)
    Cbar_k += tmp
    es[1] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2

    tmp = mf.apply_ppnl_kpt(C_k, kpt, mesh=mesh, Gv=Gv)
    Cbar_k += tmp
    es[2] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2

    if ret_E:
        return Cbar_k, es
    else:
        return Cbar_k


def apply_Fock_kpt(mf, C_k, kpt, C_ks, cell=None, kpts=None,
                   vpplocR=None, vj_R=None, exxdiv=None, ret_E=False):
    r''' Apply Fock operator to orbitals at given k-point
        Math:
            |psibar_ik> = hat{F} |psi_ik>, for all i and k = kpt
        Note:
            The Fock operator is computed using C_ks and applied to C_k, which does NOT have to be the same as C_ks[k].
    '''
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts
    if vpplocR is None: vpplocR = mf.get_vpplocR()
    if vj_R is None: vj_R = mf.get_vj_R(C_ks)
    if exxdiv is None: exxdiv = mf.exxdiv

    mesh = cell.mesh
    Gv = cell.get_Gv(mesh)

    es = np.zeros([5], dtype=np.complex128)

    tmp = mf.apply_kin_kpt(C_k, kpt, mesh=mesh, Gv=Gv)
    Cbar_k = tmp
    es[0] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2

    tmp = mf.apply_ppl_kpt(C_k, kpt, mesh=mesh, Gv=Gv, vpplocR=vpplocR)
    Cbar_k += tmp
    es[1] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2

    tmp = mf.apply_ppnl_kpt(C_k, kpt, mesh=mesh, Gv=Gv)
    Cbar_k += tmp
    es[2] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2

    tmp = mf.apply_vj_kpt(C_k, kpt, mesh=mesh, Gv=Gv, vj_R=vj_R)
    Cbar_k += tmp * 2
    es[3] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2.

    tmp = mf.apply_vk_kpt(C_k, kpt, C_ks, kpts, mesh=mesh, Gv=Gv, exxdiv=exxdiv)
    if exxdiv == "ewald":
        tmp += mf.madelung * C_k
    Cbar_k -= tmp
    es[4] = -np.einsum("ig,ig->", C_k.conj(), tmp)

    if ret_E:
        return Cbar_k, es
    else:
        return Cbar_k


def get_mo_energy(mf, C_ks, vpplocR=None, vj_R=None):
    if vpplocR is None: vpplocR = mf.get_vpplocR()
    if vj_R is None: vj_R = mf.get_vj_R(C_ks)

    kpts = mf.kpts
    nkpts = len(kpts)
    moe_ks = [None] * nkpts
    for k in range(nkpts):
        C_k = C_ks[k]
        Cbar_k = mf.apply_Fock_kpt(C_k, kpts[k], C_ks,
                                   vpplocR=vpplocR, vj_R=vj_R)
        moe_ks[k] = np.einsum("ig,ig->i", C_k.conj(), Cbar_k)

    return moe_ks


def energy_elec(mf, C_ks, moe_ks=None, vpplocR=None, vj_R=None):
    ''' Compute the electronic energy
    Pass `moe_ks` to avoid the cost of applying the expensive vj and vk.
    '''
    if vpplocR is None: vpplocR = mf.get_vpplocR()

    kpts = mf.kpts
    nkpts = len(kpts)
    e_ks = np.zeros(nkpts, dtype=np.complex128)
    if moe_ks is None:
        if vj_R is None: vj_R = mf.get_vj_R(C_ks)
        for k in range(nkpts):
            e_comp = mf.apply_Fock_kpt(C_ks[k], kpts[k], C_ks,
                                      vpplocR=vpplocR, vj_R=vj_R, ret_E=True)[1]
            e_ks[k] = np.sum(e_comp)
    else:
        for k in range(nkpts):
            e1_comp = mf.apply_h1e_kpt(C_ks[k], kpts[k], vpplocR=vpplocR,
                                       ret_E=True)[1]
            e_ks[k] = np.sum(e1_comp) * 0.5 + np.sum(moe_ks[k])

    e_scf = np.sum(e_ks) / nkpts

    return e_scf


def energy_tot(mf, C_ks, moe_ks=None, vpplocR=None, vj_R=None):
    e_nuc = mf.cell.energy_nuc()
    e_scf = mf.energy_elec(C_ks, moe_ks=moe_ks, vpplocR=vpplocR, vj_R=vj_R)
    e_tot = e_scf + e_nuc
    return e_tot


def converge_band_kpt(mf, C_k, kpt, C_ks, kpts,
                      moe_k=None, vpplocR=None, vj_R=None,
                      conv_tol_davidson=1e-6,
                      max_cycle_davidson=100,
                      verbose_davidson=0):
    ''' Converge all occupied orbitals for a given k-point using davidson algorithm
    '''

    fc = [0]
    def FC(C_k_, ret_E=False):
        fc[0] += 1
        C_k_ = np.asarray(C_k_)
        Cbar_k_ = mf.apply_Fock_kpt(C_k_, kpt, C_ks, vpplocR=vpplocR, vj_R=vj_R)
        return Cbar_k_

    if moe_k is None:
        dF = np.einsum("ig,ig->g", C_k.conj(), FC(C_k))
    else:
        dF = np.einsum("ig,ig->g", C_k.conj(), moe_k.reshape(-1,1)*C_k)
    precond = lambda dx, e, x0: dx/(dF - e)

    no_k = C_k.shape[0]

    conv, e, c = lib.davidson1(FC, C_k, precond,
                               nroots=no_k,
                               verbose=verbose_davidson,
                               tol=conv_tol_davidson,
                               max_cycle=max_cycle_davidson)
    c = np.asarray(c)

    return conv, e, c, fc[0]


def converge_band(mf, C_ks, kpts, Cout_ks=None,
                  moe_ks=None, vpplocR=None, vj_R=None,
                  conv_tol_davidson=1e-6,
                  max_cycle_davidson=100,
                  verbose_davidson=0):
    if vpplocR is None: vpplocR = mf.get_vpplocR()
    if vj_R is None: vj_R = mf.get_vj_R(C_ks)

    nkpts = len(kpts)
    if Cout_ks is None: Cout_ks = [None] * nkpts
    conv_ks = [None] * nkpts
    moeout_ks = [None] * nkpts
    fc_ks = [None] * nkpts

    for k in range(nkpts):
        conv_, moeout_ks[k], Cout_ks[k], fc_ks[k] = \
                    mf.converge_band_kpt(C_ks[k], kpts[k], C_ks, kpts,
                                         moe_k=moe_ks[k],
                                         vpplocR=vpplocR, vj_R=vj_R,
                                         conv_tol_davidson=conv_tol_davidson,
                                         max_cycle_davidson=max_cycle_davidson,
                                         verbose_davidson=verbose_davidson)
        conv_ks[k] = np.prod(conv_)

    return conv_ks, moeout_ks, Cout_ks, fc_ks


class PWKRHF(lib.StreamObject):
    '''PWKRHF base class. non-relativistic RHF using PW basis.
    '''

    conv_tol = getattr(__config__, 'pbc_pwscf_krhf_PWKRHF_conv_tol', 1e-6)
    conv_tol_davidson = getattr(__config__,
                                'pbc_pwscf_krhf_PWKRHF_conv_tol_davidson', 1e-7)
    max_cycle = getattr(__config__, 'pbc_pwscf_krhf_PWKRHF_max_cycle', 100)
    max_cycle_davidson = getattr(__config__,
                                 'pbc_pwscf_krhf_PWKRHF_max_cycle_davidson', 30)
    verbose_davidson = getattr(__config__,
                               'pbc_pwscf_krhf_PWKRHF_verbose_davidson', 0)
    conv_check = getattr(__config__, 'scf_hf_SCF_conv_check', True)
    check_convergence = None
    callback = None

    def __init__(self, cell, kpts=np.zeros((1,3)), ekincut=None,
        exxdiv=getattr(__config__, 'pbc_scf_PWKRHF_exxdiv', 'ewald')):

        if not cell._built:
            sys.stderr.write('Warning: cell.build() is not called in input\n')
            cell.build()

        self.cell = cell
        mol_hf.SCF.__init__(self, cell)

        self.kpts = kpts
        self.exxdiv = exxdiv
        if self.exxdiv == "ewald":
            self.madelung = tools.pbc.madelung(self.cell, self.kpts)

        self.exx_built = False
        self._keys = self._keys.union(['cell', 'exx_built', 'exxdiv'])

    def dump_flags(self):

        logger.info(self, '******** PBC PWSCF flags ********')
        logger.info(self, "ke_cutoff = %s", self.cell.ke_cutoff)
        logger.info(self, "mesh = %s (%d PWs)", self.cell.mesh,
                    np.prod(self.cell.mesh))
        logger.info(self, "SCF conv_tol = %s", self.conv_tol)
        logger.info(self, "SCF max_cycle = %d", self.max_cycle)
        logger.info(self, "Davidson conv_tol = %s", self.conv_tol_davidson)
        logger.info(self, "Davidson max_cycle = %d", self.max_cycle_davidson)
        if self.chkfile:
            logger.info(self, 'chkfile to save SCF result = %s', self.chkfile)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])

        logger.info(self, 'kpts = %s', self.kpts)
        logger.info(self, 'Exchange divergence treatment (exxdiv) = %s',
                    self.exxdiv)

        cell = self.cell
        if ((cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum') and
            isinstance(self.exxdiv, str) and self.exxdiv.lower() == 'ewald'):
            madelung = self.madelung
            logger.info(self, '    madelung (= occupied orbital energy shift) = %s', madelung)
            logger.info(self, '    Total energy shift due to Ewald probe charge'
                        ' = -1/2 * Nelec*madelung = %.12g',
                        madelung*cell.nelectron * -.5)

    def dump_chk(self, envs):
        if self.chkfile:
            no_ks = np.asarray([C0.shape[0] for C0 in envs['C0_ks']])
            mo_occ = [np.asarray([1]*no) for no in no_ks]
            chkfile.dump_scf(self.mol, self.chkfile,
                             envs['e_tot'], envs['moe_ks'],
                             envs['C0_ks'], mo_occ,
                             overwrite_mol=False)
        return self

    def kernel(self, **kwargs):
        self.dump_flags()
        kernel(self, kpts, C0_ks=None,
               conv_tol=mf.conv_tol, max_cycle=mf.max_cycle,
               conv_tol_davidson=mf.conv_tol_davidson,
               max_cycle_davidson=mf.max_cycle_davidson,
               verbose_davidson=mf.verbose_davidson,
               conv_check=self.conv_check,
               callback=self.callback,
               **kwargs)

    dump_moe = dump_moe
    get_init_guess = get_init_guess
    get_mo_energy = get_mo_energy
    apply_h1e_kpt = apply_h1e_kpt
    apply_Fock_kpt = apply_Fock_kpt
    energy_elec = energy_elec
    energy_tot = energy_tot
    converge_band_kpt = converge_band_kpt
    converge_band = converge_band
    apply_kin_kpt = pw_helper.apply_kin_kpt
    apply_ppl_kpt = pw_helper.apply_ppl_kpt
    apply_ppnl_kpt = pw_helper.apply_ppnl_kpt
    apply_vj_kpt = pw_helper.apply_vj_kpt
    apply_vk_kpt = pw_helper.apply_vk_kpt
    get_vpplocR = pw_helper.get_vpplocR
    get_vj_R = pw_helper.get_vj_R


if __name__ == "__main__":
    cell = gto.Cell(
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994",
        a = np.asarray(
            [[0.       , 1.78339987, 1.78339987],
            [1.78339987, 0.        , 1.78339987],
            [1.78339987, 1.78339987, 0.        ]]),
        basis="gth-szv",
        ke_cutoff=100,
        pseudo="gth-pade",
    )
    cell.build()
    cell.verbose = 6

    nk = 2
    kmesh = (nk,)*3
    kpts = cell.make_kpts(kmesh)

    mf = PWKRHF(cell, kpts)
    mf.kernel()
