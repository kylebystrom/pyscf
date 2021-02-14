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
        C_ks = mf.get_init_guess(key=mf.init_guess)
    else:
        C_ks = C0_ks

    # init E
    vpplocR = mf.get_vpplocR()
    vj_R = mf.get_vj_R(C_ks)
    moe_ks = mf.get_mo_energy(C_ks, vpplocR=vpplocR, vj_R=vj_R)
    e_tot = mf.energy_tot(C_ks, moe_ks=moe_ks, vpplocR=vpplocR)
    logger.info(mf, 'init E= %.15g', e_tot)
    mf.dump_moe(moe_ks)

    scf_conv = False

    if mf.max_cycle <= 0:
        return scf_conv, e_tot, moe_ks, C_ks

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(cell, mf.chkfile)

    fc_tot = 0
    fc_this = 0
    cput1 = logger.timer(mf, 'initialize pwscf', *cput0)
    for cycle in range(max_cycle):

        if cycle > 0:   # already built for init E at cycle 0
            vj_R = mf.get_vj_R(C_ks)

        conv_ks, moe_ks, C_ks, fc_ks = mf.converge_band(
                            C_ks, kpts, moe_ks=moe_ks,
                            vpplocR=vpplocR, vj_R=vj_R,
                            conv_tol_davidson=conv_tol_davidson,
                            max_cycle_davidson=max_cycle_davidson,
                            verbose_davidson=verbose_davidson)
        fc_this = sum(fc_ks)
        fc_tot += fc_this

        last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, moe_ks=moe_ks, vpplocR=vpplocR)
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
        conv_ks, moe_ks, C_ks, fc_ks = mf.converge_band(
                            C_ks, kpts, moe_ks=moe_ks,
                            vpplocR=vpplocR, vj_R=vj_R,
                            conv_tol_davidson=conv_tol_davidson,
                            max_cycle_davidson=max_cycle_davidson,
                            verbose_davidson=verbose_davidson)
        fc_this = sum(fc_ks)
        fc_tot += fc_this
        last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, moe_ks=moe_ks, vpplocR=vpplocR)
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
    return scf_conv, e_tot, moe_ks, C_ks


def kernel_doubleloop(
            mf, kpts, C0_ks=None, conv_tol=1.E-6, conv_tol_davidson=1.E-6,
            max_cycle=100, max_cycle_davidson=10, verbose_davidson=0,
            ace_exx=True, damp_type="anderson", damp_factor=0.3,
            dump_chk=True, conv_check=True, callback=None, **kwargs):
    ''' Kernel function for SCF in a PW basis
        Note:
            This double-loop implementation follows closely the implementation in Quantum ESPRESSO.
    '''

    cput0 = (time.clock(), time.time())

    cell = mf.cell
    nkpts = len(kpts)

    if C0_ks is None:
        C_ks = mf.get_init_guess(key=mf.init_guess)
    else:
        C_ks = C0_ks

    # init E
    mesh = cell.mesh
    Gv = cell.get_Gv(mesh)
    vpplocR = mf.get_vpplocR(mesh=mesh, Gv=Gv)
    vj_R = mf.get_vj_R(C_ks, mesh=mesh, Gv=Gv)
    ace_xi_ks = pw_helper.initialize_ACE(mf, C_ks, ace_exx, mesh=mesh, Gv=Gv)
    C_ks_exx = list(C_ks) if ace_xi_ks is None else None
    moe_ks = mf.get_mo_energy(C_ks, C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                              vpplocR=vpplocR, vj_R=vj_R)
    e_tot = mf.energy_tot(C_ks, moe_ks=moe_ks, vpplocR=vpplocR)
    logger.info(mf, 'init E= %.15g', e_tot)
    mf.dump_moe(moe_ks)

    scf_conv = False

    if mf.max_cycle <= 0:
        return scf_conv, e_tot, moe_ks, C_ks

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(cell, mf.chkfile)

    fc_tot = 0
    fc_this = 0
    cput1 = logger.timer(mf, 'initialize pwscf', *cput0)

    # chg_conv_tol = 1e-2
    chg_conv_tol = 0.1
    for cycle in range(max_cycle):

        if cycle > 0:
            chg_conv_tol = min(chg_conv_tol, max(conv_tol, 0.1*abs(de)))
        conv_tol_davidson = max(conv_tol*0.1, chg_conv_tol*0.01)
        logger.debug(mf, "  Performing charge SCF with conv_tol= %.3g conv_tol_davidson= %.3g", chg_conv_tol, conv_tol_davidson)

        # charge SCF
        chg_scf_conv, fc_this, C_ks, moe_ks, e_tot = mf.kernel_charge(
                                C_ks, kpts, mesh=mesh, Gv=Gv,
                                max_cycle=max_cycle, conv_tol=chg_conv_tol,
                                max_cycle_davidson=max_cycle_davidson,
                                conv_tol_davidson=conv_tol_davidson,
                                verbose_davidson=verbose_davidson,
                                damp_type=damp_type, damp_factor=damp_factor,
                                moe_ks=moe_ks,
                                C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                                vpplocR=vpplocR, vj_R=vj_R,
                                last_hf_e=e_tot)
        fc_tot += fc_this
        if not chg_scf_conv:
            logger.warn(mf, "  Charge SCF not converged.")

        # update coulomb potential, ace_xi_ks, and energies
        vj_R = mf.get_vj_R(C_ks)
        ace_xi_ks = pw_helper.initialize_ACE(mf, C_ks, ace_exx,
                                             mesh=mesh, Gv=Gv)
        C_ks_exx = list(C_ks) if ace_xi_ks is None else None
        moe_ks = mf.get_mo_energy(C_ks, C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                                  vpplocR=vpplocR, vj_R=vj_R)
        last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                              vpplocR=vpplocR, vj_R=vj_R)
        de = e_tot - last_hf_e

        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  %d FC (%d tot)',
                    cycle+1, e_tot, de, fc_this, fc_tot)
        mf.dump_moe(moe_ks)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(de) < conv_tol:
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
        chg_conv_tol = min(chg_conv_tol, max(conv_tol, 0.1*abs(de)))
        conv_tol_davidson = max(conv_tol*0.1, chg_conv_tol*0.01)
        logger.debug(mf, "  Performing charge SCF with conv_tol= %.3g conv_tol_davidson= %.3g", chg_conv_tol, conv_tol_davidson)

        chg_scf_conv, fc_this, C_ks, moe_ks, e_tot = mf.kernel_charge(
                                C_ks, kpts, mesh=mesh, Gv=Gv,
                                max_cycle=max_cycle, conv_tol=chg_conv_tol,
                                max_cycle_davidson=max_cycle_davidson,
                                conv_tol_davidson=conv_tol_davidson,
                                verbose_davidson=verbose_davidson,
                                damp_type=damp_type, damp_factor=damp_factor,
                                moe_ks=moe_ks,
                                C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                                vpplocR=vpplocR, vj_R=vj_R,
                                last_hf_e=e_tot)
        fc_tot += fc_this
        vj_R = mf.get_vj_R(C_ks)
        ace_xi_ks = pw_helper.initialize_ACE(mf, C_ks, ace_exx,
                                             mesh=mesh, Gv=Gv)
        C_ks_exx = list(C_ks) if ace_xi_ks is None else None
        moe_ks = mf.get_mo_energy(C_ks, C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                                  vpplocR=vpplocR, vj_R=vj_R)
        last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                              vpplocR=vpplocR, vj_R=vj_R)
        de = e_tot - last_hf_e

        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  %d FC (%d tot)',
                    e_tot, de, fc_this, fc_tot)
        mf.dump_moe(moe_ks)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(de) < conv_tol:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel(locals())
    return scf_conv, e_tot, moe_ks, C_ks


def kernel_charge(mf, C_ks, kpts, mesh=None, Gv=None,
                  max_cycle=50, conv_tol=1e-6,
                  max_cycle_davidson=10, conv_tol_davidson=1e-8,
                  verbose_davidson=0,
                  damp_type="anderson", damp_factor=0.3,
                  moe_ks=None,
                  C_ks_exx=None, ace_xi_ks=None,
                  vpplocR=None, vj_R=None,
                  last_hf_e=None):

    cell = mf.cell
    if vpplocR is None: vpplocR = mf.get_vpplocR()
    if vj_R is None: vj_R = mf.get_vj_R(C_ks)
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)

    scf_conv = False

    fc_tot = 0

    if damp_type.lower() == "simple":
        chgmixer = pw_helper.SimpleMixing(mf, damp_factor)
    elif damp_type.lower() == "anderson":
        chgmixer = pw_helper.AndersonMixing(mf)

    cput1 = (time.clock(), time.time())
    for cycle in range(max_cycle):

        if cycle > 0:   # charge mixing
            vj_R = chgmixer.next_step(mf, vj_R, vj_R-last_vj_R)

        conv_ks, moe_ks, C_ks, fc_ks = mf.converge_band(
                            C_ks, kpts, mesh=mesh, Gv=Gv,
                            moe_ks=moe_ks,
                            C_ks_exx=C_ks_exx,
                            ace_xi_ks=ace_xi_ks,
                            vpplocR=vpplocR, vj_R=vj_R,
                            conv_tol_davidson=conv_tol_davidson,
                            max_cycle_davidson=max_cycle_davidson,
                            verbose_davidson=verbose_davidson)
        fc_this = sum(fc_ks)
        fc_tot += fc_this

        # update coulomb potential and energy
        last_vj_R = vj_R
        vj_R = mf.get_vj_R(C_ks)

        if cycle > 0: last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                              vpplocR=vpplocR, vj_R=vj_R)
        if not last_hf_e is None:
            de = e_tot-last_hf_e
        else:
            de = float("inf")
        logger.debug(mf, '  chg cyc= %d E= %.15g  delta_E= %4.3g  %d FC (%d tot)',
                    cycle+1, e_tot, de, fc_this, fc_tot)
        mf.dump_moe(moe_ks, trigger_level=logger.DEBUG3)

        if abs(de) < conv_tol:
            scf_conv = True

        cput1 = logger.timer_debug1(mf, 'chg cyc= %d'%(cycle+1),
                                    *cput1)

        if scf_conv:
            break

    return scf_conv, fc_tot, C_ks, moe_ks, e_tot


def dump_moe(mf, moe_ks, trigger_level=logger.DEBUG):
    if mf.verbose >= trigger_level:
        np.set_printoptions(threshold=len(moe_ks))
        logger.debug(mf, '     k-point                  mo_energy')
        for k,kpt in enumerate(mf.cell.get_scaled_kpts(mf.kpts)):
            logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                         k, kpt[0], kpt[1], kpt[2], moe_ks[k].real)
        np.set_printoptions(threshold=1000)


def get_init_guess(mf, key="hcore"):
    tick = np.asarray([time.clock(), time.time()])

    cell = mf.cell
    kpts = mf.kpts
    nkpts = len(kpts)

    logger.info(mf, "generating init guess using %s basis", cell.basis)

    verbose = cell.verbose
    cell.verbose = 0
    if len(kpts) < 30:
        pmf = scf.KRHF(cell, kpts)
    else:
        pmf = scf.KRHF(cell, kpts).density_fit()
    cell.verbose = verbose
    pmf.exxdiv = mf.exxdiv

    if key.lower() == "cycle1":
        pmf.max_cycle = 0
        pmf.kernel()
        mo_coeff = pmf.mo_coeff
        mo_occ = pmf.mo_occ
    elif key.lower() in ["hcore", "h1e"]:
        h1e = pmf.get_hcore()
        s1e = pmf.get_ovlp()
        mo_energy, mo_coeff = pmf.eig(h1e, s1e)
        mo_occ = pmf.get_occ(mo_energy, mo_coeff)
    else:
        raise NotImplementedError("Init guess %s not implemented" % key)

    Co_ks = pw_helper.get_Co_ks_G(cell, kpts, mo_coeff, mo_occ)
    for i,kpt in enumerate(kpts):
        Sk = Co_ks[i].conj() @ Co_ks[i].T
        nonorth_err = np.max(np.abs(Sk - np.eye(Sk.shape[0])))
        if nonorth_err > mf.conv_tol * 1e-3:
            logger.warn(mf, "non-orthogonality detected in the initial MOs (max |off-diag ovlp|= %s) for kpt %d. Symm-orth them now.", nonorth_err, i)
        e, u = scipy.linalg.eigh(Sk)
        Co_ks[i] = (u*e**-0.5).T @ Co_ks[i]

    tock = np.asarray([time.clock(), time.time()])
    key = "t-init"
    if not key in mf.scf_summary:
        mf.scf_summary[key] = np.zeros(2)
    mf.scf_summary[key] += tock - tick

    return Co_ks


def apply_h1e_kpt(mf, C_k, kpt, mesh=None, Gv=None, vpplocR=None, ret_E=False):
    r''' Apply 1e part of the Fock operator to orbitals at given k-points.
        Math:
            |psibar_ik> = (hat{T} + hat{vpp}) |psi_ik>, for all i and k = kpt
    '''
    cell = mf.cell
    if vpplocR is None: vpplocR = mf.get_vpplocR()
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)

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
        if (np.abs(es.imag) > 1e-6).any():
            e_comp = mf.scf_summary["e_comp_name_lst"][:3]
            icomps = np.where(np.abs(es.imag) > 1e-6)[0]
            logger.warn(mf, "Energy has large imaginary part:" +
                     "%s : %s\n" * len(icomps),
                     *[s for i in icomps for s in [e_comp[i],es[i]]])
        es = es.real

        return Cbar_k, es
    else:
        return Cbar_k


def apply_Fock_kpt(mf, C_k, kpt, C_ks, cell=None, kpts=None, mesh=None, Gv=None,
                   C_ks_exx=None, ace_xi_k=None,
                   vpplocR=None, vj_R=None, exxdiv=None, ret_E=False):
    r''' Apply Fock operator to orbitals at given k-point
        Math:
            |psibar_ik> = hat{F} |psi_ik>, for all i and k = kpt
        Note:
            1. The Fock operator is computed using C_ks and applied to C_k, which does NOT have to be the same as C_ks[k].
            2. If C_ks_exx is given, the EXX potential will be evaluated using C_ks_exx. This is useful in the double-loop formulation of SCF.
    '''
    if cell is None: cell = mf.cell
    if kpts is None: kpts = mf.kpts
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    if vpplocR is None: vpplocR = mf.get_vpplocR()
    if vj_R is None: vj_R = mf.get_vj_R(C_ks)
    if exxdiv is None: exxdiv = mf.exxdiv

    es = np.zeros([5], dtype=np.complex128)

    tspans = np.zeros((5,2))
    tick = np.asarray([time.clock(), time.time()])

    tmp = mf.apply_kin_kpt(C_k, kpt, mesh=mesh, Gv=Gv)
    Cbar_k = tmp
    es[0] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2
    tock = np.asarray([time.clock(), time.time()])
    tspans[0] = tock - tick

    tmp = mf.apply_ppl_kpt(C_k, kpt, mesh=mesh, Gv=Gv, vpplocR=vpplocR)
    Cbar_k += tmp
    es[1] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2
    tick = np.asarray([time.clock(), time.time()])
    tspans[1] = tick - tock

    tmp = mf.apply_ppnl_kpt(C_k, kpt, mesh=mesh, Gv=Gv)
    Cbar_k += tmp
    es[2] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2
    tock = np.asarray([time.clock(), time.time()])
    tspans[2] = tock - tick

    tmp = mf.apply_vj_kpt(C_k, kpt, mesh=mesh, Gv=Gv, vj_R=vj_R)
    Cbar_k += tmp * 2
    es[3] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2.
    tick = np.asarray([time.clock(), time.time()])
    tspans[3] = tick - tock

    if C_ks_exx is None: C_ks_exx = C_ks
    tmp = mf.apply_vk_kpt(C_k, kpt, C_ks_exx, kpts, ace_xi_k=ace_xi_k,
                          mesh=mesh, Gv=Gv, exxdiv=exxdiv)
    if exxdiv == "ewald":
        tmp += mf.madelung * C_k
    Cbar_k -= tmp
    es[4] = -np.einsum("ig,ig->", C_k.conj(), tmp)
    tock = np.asarray([time.clock(), time.time()])
    tspans[4] = tock - tick

    for icomp,comp in enumerate(mf.scf_summary["e_comp_name_lst"]):
        key = "t-%s" % comp
        if not key in mf.scf_summary:
            mf.scf_summary[key] = np.zeros(2)
        mf.scf_summary[key] += tspans[icomp]

    if ret_E:
        if (np.abs(es.imag) > 1e-6).any():
            e_comp = mf.scf_summary["e_comp_name_lst"]
            icomps = np.where(np.abs(es.imag) > 1e-6)[0]
            logger.warn(mf, "Energy has large imaginary part:" +
                     "%s : %s\n" * len(icomps),
                     *[s for i in icomps for s in [e_comp[i],es[i]]])
        es = es.real

        return Cbar_k, es
    else:
        return Cbar_k


def get_mo_energy(mf, C_ks, mesh=None, Gv=None, C_ks_exx=None, ace_xi_ks=None,
                  vpplocR=None, vj_R=None):
    cell = mf.cell
    if vpplocR is None: vpplocR = mf.get_vpplocR()
    if vj_R is None: vj_R = mf.get_vj_R(C_ks)
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)

    kpts = mf.kpts
    nkpts = len(kpts)
    moe_ks = [None] * nkpts
    for k in range(nkpts):
        C_k = C_ks[k]
        ace_xi_k = None if ace_xi_ks is None else ace_xi_ks[k]
        Cbar_k = mf.apply_Fock_kpt(C_k, kpts[k], C_ks, mesh=mesh, Gv=Gv,
                                   C_ks_exx=C_ks_exx, ace_xi_k=ace_xi_k,
                                   vpplocR=vpplocR, vj_R=vj_R)
        moe_k = np.einsum("ig,ig->i", C_k.conj(), Cbar_k)
        if (moe_k.imag > 1e-6).any():
            logger.warn(mf, "MO energies have imaginary part %s for kpt %d",
                        moe_k, k)
        moe_ks[k] = moe_k.real

    return moe_ks


def energy_elec(mf, C_ks, mesh=None, Gv=None, moe_ks=None,
                C_ks_exx=None, ace_xi_ks=None,
                vpplocR=None, vj_R=None):
    ''' Compute the electronic energy
    Pass `moe_ks` to avoid the cost of applying the expensive vj and vk.
    '''
    cell = mf.cell
    if vpplocR is None: vpplocR = mf.get_vpplocR()
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)

    kpts = mf.kpts
    nkpts = len(kpts)
    e_ks = np.zeros(nkpts)
    if moe_ks is None:
        if vj_R is None: vj_R = mf.get_vj_R(C_ks)
        e_comp = np.zeros(5)
        for k in range(nkpts):
            ace_xi_k = None if ace_xi_ks is None else ace_xi_ks[k]
            e_comp_k = mf.apply_Fock_kpt(C_ks[k], kpts[k], C_ks,
                                         mesh=mesh, Gv=Gv,
                                         C_ks_exx=C_ks_exx,
                                         ace_xi_k=ace_xi_k,
                                         vpplocR=vpplocR, vj_R=vj_R,
                                         ret_E=True)[1]
            e_ks[k] = np.sum(e_comp_k)
            e_comp += e_comp_k
        e_comp /= nkpts
        for comp,e in zip(mf.scf_summary["e_comp_name_lst"],e_comp):
            mf.scf_summary[comp] = e
    else:
        for k in range(nkpts):
            e1_comp = mf.apply_h1e_kpt(C_ks[k], kpts[k], mesh=mesh, Gv=Gv,
                                       vpplocR=vpplocR, ret_E=True)[1]
            e_ks[k] = np.sum(e1_comp) * 0.5 + np.sum(moe_ks[k])
    e_scf = np.sum(e_ks) / nkpts

    return e_scf


def energy_tot(mf, C_ks, moe_ks=None, C_ks_exx=None, ace_xi_ks=None,
               vpplocR=None, vj_R=None):
    e_nuc = mf.scf_summary["nuc"]
    e_scf = mf.energy_elec(C_ks, moe_ks=moe_ks,
                           C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                           vpplocR=vpplocR, vj_R=vj_R)
    e_tot = e_scf + e_nuc
    return e_tot


def converge_band_kpt(mf, C_k, kpt, C_ks, kpts, mesh=None, Gv=None,
                      moe_k=None, C_ks_exx=None,
                      ace_xi_k=None,
                      vpplocR=None, vj_R=None,
                      conv_tol_davidson=1e-6,
                      max_cycle_davidson=100,
                      verbose_davidson=0):
    ''' Converge all occupied orbitals for a given k-point using davidson algorithm
    '''

    fc = [0]
    def FC(C_k_, ret_E=False):
        fc[0] += 1
        C_k_ = np.asarray(C_k_)
        Cbar_k_ = mf.apply_Fock_kpt(C_k_, kpt, C_ks, mesh=mesh, Gv=Gv,
                                    C_ks_exx=C_ks_exx, ace_xi_k=ace_xi_k,
                                    vpplocR=vpplocR, vj_R=vj_R)
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


def converge_band(mf, C_ks, kpts, Cout_ks=None, mesh=None, Gv=None,
                  C_ks_exx=None, ace_xi_ks=None,
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
        ace_xi_k = None if ace_xi_ks is None else ace_xi_ks[k]
        conv_, moeout_ks[k], Cout_ks[k], fc_ks[k] = \
                    mf.converge_band_kpt(C_ks[k], kpts[k], C_ks, kpts,
                                         mesh=mesh, Gv=Gv,
                                         moe_k=moe_ks[k],
                                         C_ks_exx=C_ks_exx,
                                         ace_xi_k=ace_xi_k,
                                         vpplocR=vpplocR, vj_R=vj_R,
                                         conv_tol_davidson=conv_tol_davidson,
                                         max_cycle_davidson=max_cycle_davidson,
                                         verbose_davidson=verbose_davidson)
        conv_ks[k] = np.prod(conv_)

    return conv_ks, moeout_ks, Cout_ks, fc_ks


# class PWKRHF(lib.StreamObject):
class PWKRHF(mol_hf.SCF):
    '''PWKRHF base class. non-relativistic RHF using PW basis.
    '''

    conv_tol = getattr(__config__, 'pbc_pwscf_krhf_PWKRHF_conv_tol', 1e-6)
    conv_tol_davidson = getattr(__config__,
                                'pbc_pwscf_krhf_PWKRHF_conv_tol_davidson', 1e-7)
    max_cycle = getattr(__config__, 'pbc_pwscf_krhf_PWKRHF_max_cycle', 100)
    max_cycle_davidson = getattr(__config__,
                                 'pbc_pwscf_krhf_PWKRHF_max_cycle_davidson',
                                 100)
    verbose_davidson = getattr(__config__,
                               'pbc_pwscf_krhf_PWKRHF_verbose_davidson', 0)
    double_loop_scf = getattr(__config__,
                              'pbc_pwscf_krhf_PWKRHF_double_loop_scf', True)
    ace_exx = getattr(__config__, 'pbc_pwscf_krhf_PWKRHF_ace_exx', True)
    damp_type = getattr(__config__, 'pbc_pwscf_krhf_PWKRHF_damp_type',
                        "anderson")
    damp_factor = getattr(__config__, 'pbc_pwscf_krhf_PWKRHF_damp_factor', 0.3)
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
        self.scf_summary["nuc"] = self.cell.energy_nuc()
        self.scf_summary["e_comp_name_lst"] = ["kin", "ppl", "ppnl", "coul", "ex"]

        self.init_guess = "hcore"

        self.exx_built = False
        self._keys = self._keys.union(['cell', 'exx_built', 'exxdiv'])

    def dump_flags(self):

        logger.info(self, '******** PBC PWSCF flags ********')
        logger.info(self, "ke_cutoff = %s", self.cell.ke_cutoff)
        logger.info(self, "mesh = %s (%d PWs)", self.cell.mesh,
                    np.prod(self.cell.mesh))
        logger.info(self, "SCF init guess = %s", self.init_guess)
        logger.info(self, "SCF conv_tol = %s", self.conv_tol)
        logger.info(self, "SCF max_cycle = %d", self.max_cycle)
        logger.info(self, "Davidson conv_tol = %s", self.conv_tol_davidson)
        logger.info(self, "Davidson max_cycle = %d", self.max_cycle_davidson)
        logger.info(self, "Use double-loop scf = %s", self.double_loop_scf)
        if self.double_loop_scf:
            logger.info(self, "Use ACE = %s", self.ace_exx)
            logger.info(self, "Damping method = %s", self.damp_type)
            if self.damp_type.lower() == "simple":
                logger.info(self, "Damping factor = %s", self.damp_factor)
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
            no_ks = np.asarray([C.shape[0] for C in envs['C_ks']])
            mo_occ = [np.asarray([1]*no) for no in no_ks]
            chkfile.dump_scf(self.mol, self.chkfile,
                             envs['e_tot'], envs['moe_ks'],
                             envs['C_ks'], mo_occ,
                             overwrite_mol=False)
        return self

    def dump_scf_summary(self, verbose=logger.DEBUG):
        log = logger.new_logger(self, verbose)
        summary = self.scf_summary
        def write(fmt, key):
            if key in summary:
                log.info(fmt, summary[key])
        log.info('**** SCF Summaries ****')
        log.info('Total Energy =                    %24.15f', self.e_tot)
        write('Nuclear Repulsion Energy =        %24.15f', 'nuc')
        write('Kinetic Energy =                  %24.15f', 'kin')
        write('Local PP Energy =                 %24.15f', 'ppl')
        write('Non-local PP Energy =             %24.15f', 'ppnl')
        write('Two-electron Coulomb Energy =     %24.15f', 'coul')
        write('Two-electron Exchjange Energy =   %24.15f', 'ex')
        write('Empirical Dispersion Energy =     %24.15f', 'dispersion')
        write('PCM Polarization Energy =         %24.15f', 'epcm')
        write('EFP Energy =                      %24.15f', 'efp')
        if getattr(self, 'entropy', None):
            log.info('(Electronic) Entropy              %24.15f', self.entropy)
            log.info('(Electronic) Zero Point Energy    %24.15f', self.e_zero)
            log.info('Free Energy =                     %24.15f', self.e_free)

        t_tot = np.zeros(2)
        if 't-init' in summary:
            log.info('CPU time for %10s %9.2f, wall time %9.2f',
                     "init guess".ljust(10), *summary['t-init'])
            t_tot += summary['t-init']
        if 't-ace' in summary:
            log.info('CPU time for %10s %9.2f, wall time %9.2f',
                     "init ACE".ljust(10), *summary['t-ace'])
            t_tot += summary['t-ace']
        for op in summary["e_comp_name_lst"]:
            log.info('CPU time for %10s %9.2f, wall time %9.2f',
                     ("op %s"%op).ljust(10), *summary['t-%s'%op])
            t_tot += summary['t-%s'%op]
        log.info('CPU time for %10s %9.2f, wall time %9.2f',
                 "full SCF".ljust(10), *t_tot)

    def scf(self, **kwargs):
        self.dump_flags()

        if self.double_loop_scf:
            self.converged, self.e_tot, self.mo_energy, self.mo_coeff = \
                        kernel_doubleloop(self, self.kpts, C0_ks=None,
                               conv_tol=self.conv_tol, max_cycle=self.max_cycle,
                               conv_tol_davidson=self.conv_tol_davidson,
                               max_cycle_davidson=self.max_cycle_davidson,
                               verbose_davidson=self.verbose_davidson,
                               ace_exx=self.ace_exx,
                               damp_type=self.damp_type,
                               damp_factor=self.damp_factor,
                               conv_check=self.conv_check,
                               callback=self.callback,
                               **kwargs)
        else:
            self.converged, self.e_tot, self.mo_energy, self.mo_coeff = \
                        kernel(self, self.kpts, C0_ks=None,
                               conv_tol=self.conv_tol, max_cycle=self.max_cycle,
                               conv_tol_davidson=self.conv_tol_davidson,
                               max_cycle_davidson=self.max_cycle_davidson,
                               verbose_davidson=self.verbose_davidson,
                               conv_check=self.conv_check,
                               callback=self.callback,
                               **kwargs)
        self._finalize()
        return self.e_tot
    kernel = lib.alias(scf, alias_name='kernel')

    kernel_charge = kernel_charge
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
    cell.verbose = 5

    nk = 2
    kmesh = (nk,)*3
    kpts = cell.make_kpts(kmesh)

    mf = PWKRHF(cell, kpts)
    mf.kernel()

    mf.dump_scf_summary()
