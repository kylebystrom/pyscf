""" Hartree-Fock in the Plane Wave Basis
"""


import time
import copy
import h5py
import tempfile
import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf import __config__
from pyscf.scf import hf as mol_hf
from pyscf.scf import chkfile as mol_chkfile
from pyscf.pbc.pwscf import chkfile
from pyscf.pbc import gto, scf, tools
from pyscf.pbc.pwscf import pw_helper
from pyscf.pbc.pwscf import pseudo as pw_pseudo
from pyscf.lib import logger
import pyscf.lib.parameters as param


# TODO
# 1. fractional occupation (for metals)


THR_OCC = 1E-3


def kernel_doubleloop(mf, kpts, C0=None, facexi=None,
            nbandv=0, nbandv_extra=1,
            conv_tol=1.E-6, conv_tol_davidson=1.E-6, conv_tol_band=1e-4,
            max_cycle=100, max_cycle_davidson=10, verbose_davidson=0,
            ace_exx=True, damp_type="anderson", damp_factor=0.3,
            conv_check=True, callback=None, **kwargs):
    ''' Kernel function for SCF in a PW basis
        Note:
            This double-loop implementation follows closely the implementation in Quantum ESPRESSO.

        Args:
            C0 (list of numpy arrays):
                A list of nkpts numpy arrays, each of size nocc(k) * Npw.
            facexi (None, str or h5py file):
                Specify where to store the ACE xi vectors.
                If None, a tempfile is created and discarded at the end of the calculation. If str, a h5py file will be created and saved (for later use).
            nbandv (int):
                How many virtual bands to compute? Default is zero.
            nbandv_extra (int):
                How many extra virtual bands to include to facilitate the convergence of the davidson algorithm for the highest few virtual bands? Default is 1.
    '''

    cput0 = (time.clock(), time.time())

    cell = mf.cell
    nkpts = len(kpts)

    nbando, nbandv_tot, nband, nband_tot = mf.get_nband(nbandv, nbandv_extra)
    logger.info(mf, "Num of occ bands= %s", nbando)
    logger.info(mf, "Num of vir bands= %s", nbandv)
    logger.info(mf, "Num of all bands= %s", nband)
    logger.info(mf, "Num of extra vir bands= %s", nbandv_extra)

    # init guess and SCF chkfile
    tick = np.asarray([time.clock(), time.time()])
    C_ks, mocc_ks, fchk, dump_chk = mf.get_init_guess(nvir=nbandv_tot, C0=C0)

    pp = pw_pseudo.PWPP(cell, kpts)

    tock = np.asarray([time.clock(), time.time()])
    mf.scf_summary["t-init"] = tock - tick

    """
    In swap:
        ace_xi
        C_ks_exx
    In fchk:
        C_ks
    """

    if facexi is None:
        swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        logger.info(mf, "ACE xi vectors will be written to %s", swapfile.name)
        facexi = lib.H5TmpFile(swapfile.name)
        swapfile = None
    elif isinstance(facexi, str):
        facexi = h5py.File(facexi, "w")

    if ace_exx:
        if "ace_xi_ks" in facexi: del facexi["ace_xi_ks"]
        ace_xi_ks = facexi.create_group("ace_xi_ks")
        C_ks_exx = None
    else:
        ace_xi_ks = None
        if "C_ks_exx" in facexi: del facexi["C_ks_exx"]
        C_ks_exx = facexi.create_group("C_ks_exx")

    # init E
    mesh = cell.mesh
    Gv = cell.get_Gv(mesh)
    vj_R = mf.get_vj_R(C_ks, mocc_ks, mesh=mesh, Gv=Gv)
    mf.update_subspace_vppnloc(pp, C_ks)
    mf.initialize_ACE(C_ks, mocc_ks, kpts, mesh, Gv, ace_xi_ks)
    mf.copy_C_ks(C_ks, C_ks_exx)
    moe_ks, mocc_ks = mf.get_mo_energy(C_ks, mocc_ks,
                                       C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                                       pp=pp, vj_R=vj_R)
    e_tot = mf.energy_tot(C_ks, mocc_ks, moe_ks=moe_ks, pp=pp)
    logger.info(mf, 'init E= %.15g', e_tot)
    mf.dump_moe(moe_ks, mocc_ks, nband=nband)

    scf_conv = False

    if mf.max_cycle <= 0:
        mf.remove_extra_virbands(C_ks, moe_ks, mocc_ks, nbandv_extra)
        return scf_conv, e_tot, moe_ks, C_ks, mocc_ks

    if dump_chk:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        mol_chkfile.save_mol(cell, mf.chkfile)

    fc_tot = 0
    fc_this = 0
    cput1 = logger.timer(mf, 'initialize pwscf', *cput0)

    chg_conv_tol = 0.1
    for cycle in range(max_cycle):

        if cycle > 0:
            chg_conv_tol = min(chg_conv_tol, max(conv_tol, 0.1*abs(de)))
        conv_tol_davidson = max(conv_tol*0.1, chg_conv_tol*0.01)
        logger.debug(mf, "  Performing charge SCF with conv_tol= %.3g conv_tol_davidson= %.3g", chg_conv_tol, conv_tol_davidson)

        # charge SCF
        chg_scf_conv, fc_this, C_ks, chg_moe_ks, chg_mocc_ks, chg_e_tot = \
                            mf.kernel_charge(
                                C_ks, mocc_ks, kpts, nband, mesh=mesh, Gv=Gv,
                                max_cycle=max_cycle, conv_tol=chg_conv_tol,
                                max_cycle_davidson=max_cycle_davidson,
                                conv_tol_davidson=conv_tol_davidson,
                                verbose_davidson=verbose_davidson,
                                damp_type=damp_type, damp_factor=damp_factor,
                                C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                                pp=pp, vj_R=vj_R,
                                last_hf_e=e_tot)
        fc_tot += fc_this
        if not chg_scf_conv:
            logger.warn(mf, "  Charge SCF not converged.")

        # update coulomb potential, facexi, and energies
        mocc_ks = chg_mocc_ks
        vj_R = mf.get_vj_R(C_ks, mocc_ks)
        mf.update_subspace_vppnloc(pp, C_ks)
        mf.initialize_ACE(C_ks, mocc_ks, kpts, mesh, Gv, ace_xi_ks)
        mf.copy_C_ks(C_ks, C_ks_exx)
        last_hf_moe = moe_ks
        moe_ks, mocc_ks = mf.get_mo_energy(C_ks, mocc_ks,
                                           C_ks_exx=C_ks_exx,
                                           ace_xi_ks=ace_xi_ks,
                                           pp=pp, vj_R=vj_R)
        de_band = mf.get_band_err(moe_ks, last_hf_moe, nband)
        last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, mocc_ks,
                              C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                              pp=pp, vj_R=vj_R)
        de = e_tot - last_hf_e

        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  max|dEband|= %4.3g  %d FC (%d tot)',
                    cycle+1, e_tot, de, de_band, fc_this, fc_tot)
        mf.dump_moe(moe_ks, mocc_ks, nband=nband)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(de) < conv_tol and abs(de_band) < conv_tol_band:
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

        chg_scf_conv, fc_this, C_ks, chg_moe_ks, chg_mocc_ks, chg_e_tot = \
                            mf.kernel_charge(
                                C_ks, mocc_ks, kpts, nband, mesh=mesh, Gv=Gv,
                                max_cycle=max_cycle, conv_tol=chg_conv_tol,
                                max_cycle_davidson=max_cycle_davidson,
                                conv_tol_davidson=conv_tol_davidson,
                                verbose_davidson=verbose_davidson,
                                damp_type=damp_type, damp_factor=damp_factor,
                                C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                                pp=pp, vj_R=vj_R,
                                last_hf_e=e_tot)
        fc_tot += fc_this

        mocc_ks = chg_mocc_ks
        vj_R = mf.get_vj_R(C_ks, mocc_ks)
        mf.update_subspace_vppnloc(pp, C_ks)
        mf.initialize_ACE(C_ks, mocc_ks, kpts, mesh, Gv, ace_xi_ks)
        mf.copy_C_ks(C_ks, C_ks_exx)
        last_hf_moe = moe_ks
        moe_ks, mocc_ks = mf.get_mo_energy(C_ks, mocc_ks,
                                           C_ks_exx=C_ks_exx,
                                           ace_xi_ks=ace_xi_ks,
                                           pp=pp, vj_R=vj_R)
        de_band = mf.get_band_err(moe_ks, last_hf_moe, nband)
        last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, mocc_ks,
                              C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                              pp=pp, vj_R=vj_R)
        de = e_tot - last_hf_e

        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  max|dEband|= %4.3g  %d FC (%d tot)',
                    e_tot, de, de_band, fc_this, fc_tot)
        mf.dump_moe(moe_ks, mocc_ks, nband=nband)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(de) < conv_tol and abs(de_band) < conv_tol_band:
            scf_conv = True

    # remove extra virtual bands before return
    mf.remove_extra_virbands(C_ks, moe_ks, mocc_ks, nbandv_extra)

    if dump_chk:
        mf.dump_chk(locals())

    if callable(callback):
        callback(locals())

    if dump_chk: fchk.close()
    if ace_exx: facexi.close()

    cput1 = (time.clock(), time.time())
    mf.scf_summary["t-tot"] = np.asarray(cput1) - np.asarray(cput0)
    logger.debug(mf, '    CPU time for %s %9.2f sec, wall time %9.2f sec',
                 "scf_cycle", *mf.scf_summary["t-tot"])
    # A post-processing hook before return
    mf.post_kernel(locals())
    return scf_conv, e_tot, moe_ks, C_ks, mocc_ks


# def sort_mo(mocc_ks, moe_ks, C_ks):
#     incore = isinstance(C_ks, list)
#     nkpts = len(mocc_ks)
#     for k in range(nkpts):
#         mocc = mocc_ks[k]
#         nocc = np.sum(mocc>THR_OCC)
#         if np.sum(mocc[:nocc]>THR_OCC) < nocc:
#             mocc_ks[k] = np.asarray([2 if i < nocc else 0
#                                     for i in range(mocc.size)])
#             moe = moe_ks[k]
#             order = np.argsort(moe)[::-1]
#             moe_ks[k] = moe[order]
#             if incore:
#                 C = C_ks[k]
#             else:
#                 key = "%d"%k
#                 C = C_ks[key][()]
#                 del C_ks[key]
#                 C_ks[key] = C[order]


def get_nband(mf, nbandv, nbandv_extra):
    cell = mf.cell
    nbando = cell.nelectron // 2
    nbandv_tot = nbandv + nbandv_extra
    nband = nbando + nbandv
    nband_tot = nbando + nbandv_tot

    return nbando, nbandv_tot, nband, nband_tot


def get_band_err(mf, moe_ks, last_hf_moe, nband):
    if nband == 0: return 0.

    nkpts = len(mf.kpts)
    return np.max([np.max(abs(moe_ks[k] - last_hf_moe[k])[:nband])
                  for k in range(nkpts)])


def remove_extra_virbands(mf, C_ks, moe_ks, mocc_ks, nbandv_extra):
    if nbandv_extra > 0:
        nkpts = len(moe_ks)
        for k in range(nkpts):
            moe_ks[k] = moe_ks[k][:-nbandv_extra]
            mocc_ks[k] = mocc_ks[k][:-nbandv_extra]
        if isinstance(C_ks, list):
            for k in range(nkpts):
                C_ks[k] = C_ks[k][:-nbandv_extra]
        else:
            for k in range(nkpts):
                key = "%d" % k
                C = C_ks[key][:-nbandv_extra]
                del C_ks[key]
                C_ks[key] = C


def kernel_charge(mf, C_ks, mocc_ks, kpts, nband, mesh=None, Gv=None,
                  max_cycle=50, conv_tol=1e-6,
                  max_cycle_davidson=10, conv_tol_davidson=1e-8,
                  verbose_davidson=0,
                  damp_type="anderson", damp_factor=0.3,
                  C_ks_exx=None, ace_xi_ks=None,
                  pp=None, vj_R=None,
                  last_hf_e=None):

    cell = mf.cell
    if pp is None: pp = pw_pseudo.PWPP(cell, mf.kpts, mesh=mesh)
    if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)
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
                            C_ks, mocc_ks, kpts,
                            mesh=mesh, Gv=Gv,
                            C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                            pp=pp, vj_R=vj_R,
                            conv_tol_davidson=conv_tol_davidson,
                            max_cycle_davidson=max_cycle_davidson,
                            verbose_davidson=verbose_davidson)
        fc_this = sum(fc_ks)
        fc_tot += fc_this

        # update mo occ
        mocc_ks = mf.get_mo_occ(moe_ks)

        # update coulomb potential and energy
        last_vj_R = vj_R
        vj_R = mf.get_vj_R(C_ks, mocc_ks)

        if cycle > 0: last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, mocc_ks,
                              C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                              pp=pp, vj_R=vj_R)
        if not last_hf_e is None:
            de = e_tot-last_hf_e
        else:
            de = float("inf")
        logger.debug(mf, '  chg cyc= %d E= %.15g  delta_E= %4.3g  %d FC (%d tot)',
                    cycle+1, e_tot, de, fc_this, fc_tot)
        mf.dump_moe(moe_ks, mocc_ks, nband=nband, trigger_level=logger.DEBUG3)

        if abs(de) < conv_tol:
            scf_conv = True

        cput1 = logger.timer_debug1(mf, 'chg cyc= %d'%(cycle+1),
                                    *cput1)

        if scf_conv:
            break

    return scf_conv, fc_tot, C_ks, moe_ks, mocc_ks, e_tot


def copy_C_ks(mf, C_ks, C_ks_exx):
    if C_ks_exx is None:
        return None
    else:
        nkpts = len(C_ks)
        for k in range(nkpts):
            key = "%d" % k
            C = C_ks[k] if isinstance(C_ks, list) else C_ks[key][()]
            if isinstance(C_ks_exx, list):
                C_ks_exx[k] = C.copy()
            else:
                if key in C_ks_exx: del C_ks_exx[key]
                C_ks_exx[key] = C

        return C_ks_exx


def get_mo_occ(cell, moe_ks=None, C_ks=None, nocc=None):
    if nocc is None: nocc = cell.nelectron // 2
    if not moe_ks is None:
        nkpts = len(moe_ks)
        nocc_tot = nocc * nkpts
        e_fermi = np.sort(np.concatenate(moe_ks))[nocc_tot-1]
        EPSILON = 1e-10
        mocc_ks = [None] * nkpts
        for k in range(nkpts):
            mocc_k = np.zeros(moe_ks[k].size)
            mocc_k[moe_ks[k] < e_fermi+EPSILON] = 2
            mocc_ks[k] = mocc_k
    elif not C_ks is None:
        if isinstance(C_ks, list):
            mocc_ks = [np.asarray([2 if i < nocc else 0
                       for i in range(C_k.shape[0])]) for C_k in C_ks]
        elif isinstance(C_ks, h5py.Group):
            nkpts = len(C_ks)
            mocc_ks = [np.asarray([2 if i < nocc else 0
                       for i in range(C_ks["%d"%k].shape[0])])
                       for k in range(nkpts)]
        else:
            raise RuntimeError
    else:
        raise RuntimeError

    return mocc_ks


def dump_moe(mf, moe_ks_, mocc_ks_, nband=None, trigger_level=logger.DEBUG):
    if mf.verbose >= trigger_level:
        kpts = mf.cell.get_scaled_kpts(mf.kpts)
        nkpts = len(kpts)
        if not nband is None:
            moe_ks = [moe_ks_[k][:nband] for k in range(nkpts)]
            mocc_ks = [mocc_ks_[k][:nband] for k in range(nkpts)]
        else:
            moe_ks = moe_ks_
            mocc_ks = mocc_ks_

        has_occ = np.where([(mocc_ks[k] > THR_OCC).any()
                           for k in range(nkpts)])[0]
        if len(has_occ) > 0:
            ehomo_ks = np.asarray([np.max(moe_ks[k][mocc_ks[k]>THR_OCC])
                                  for k in has_occ])
            ehomo = np.max(ehomo_ks)
            khomos = has_occ[np.where(abs(ehomo_ks-ehomo) < 1e-4)[0]]

            logger.info(mf, '  HOMO = %.15g  kpt'+' %d'*khomos.size,
                         ehomo, *khomos)

        has_vir = np.where([(mocc_ks[k] < THR_OCC).any()
                           for k in range(nkpts)])[0]
        if len(has_vir) > 0:
            elumo_ks = np.asarray([np.min(moe_ks[k][mocc_ks[k]<THR_OCC])
                                  for k in has_vir])
            elumo = np.min(elumo_ks)
            klumos = has_vir[np.where(abs(elumo_ks-elumo) < 1e-4)[0]]

            logger.info(mf, '  LUMO = %.15g  kpt'+' %d'*klumos.size,
                         elumo, *klumos)

        if len(has_occ) >0 and len(has_vir) > 0:
            logger.debug(mf, '  Egap = %.15g', elumo-ehomo)

        np.set_printoptions(threshold=len(moe_ks[0]))
        logger.debug(mf, '     k-point                  mo_energy')
        for k,kpt in enumerate(kpts):
            if mocc_ks is None:
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s',
                             k, kpt[0], kpt[1], kpt[2], moe_ks[k].real)
            else:
                logger.debug(mf, '  %2d (%6.3f %6.3f %6.3f)   %s  %s',
                             k, kpt[0], kpt[1], kpt[2],
                             moe_ks[k][mocc_ks[k]>0].real,
                             moe_ks[k][mocc_ks[k]==0].real)
        np.set_printoptions(threshold=1000)


def orth_mo1(cell, C, mocc, thr_nonorth=1e-6, thr_lindep=1e-8, follow=True):
    """ orth occupieds and virtuals separately
    """
    orth = pw_helper.orth
    Co = C[mocc>THR_OCC]
    Cv = C[mocc<THR_OCC]
    # orth occ
    if Co.shape[0] > 0:
        Co = orth(cell, Co, thr_nonorth, thr_lindep, follow)
    # project out occ from vir and orth vir
    if Cv.shape[0] > 0:
        Cv -= lib.dot(lib.dot(Cv, Co.conj().T), Co)
        Cv = orth(cell, Cv, thr_nonorth, thr_lindep, follow)

    C = np.vstack([Co,Cv])

    return C


def orth_mo(cell, C_ks, mocc_ks, thr=1e-3):
    nkpts = len(mocc_ks)
    if isinstance(C_ks, list):
        for k in range(nkpts):
            C_ks[k] = orth_mo1(cell, C_ks[k], mocc_ks[k], thr)
    elif isinstance(C_ks, h5py.Group):
        for k in range(nkpts):
            key = "%d"%k
            C_k = C_ks[key][()]
            del C_ks[key]
            C_ks[key] = orth_mo1(cell, C_k, mocc_ks[k], thr)
    else:
        raise RuntimeError

    return C_ks


def get_init_guess(cell0, kpts, basis=None, pseudo=None, nvir=0,
                   key="hcore", fC_ks=None):
    """
        Args:
            nvir (int):
                Number of virtual bands to be evaluated. Default is zero.
            fC_ks (h5py group):
                If provided, the orbitals are written to it.
    """

    if not fC_ks is None:
        assert(isinstance(fC_ks, h5py.Group))

    nkpts = len(kpts)

    if basis is None: basis = cell0.basis
    if pseudo is None: pseudo = cell0.pseudo
    cell = copy.copy(cell0)
    cell.basis = basis
    if len(cell._ecp) > 0:  # use GTH to avoid the slow init time of ECP
        cell.pseudo = "gth-pade"
        cell.ecp = cell._ecp = cell._ecpbas = None
    else:
        cell.pseudo = pseudo
    cell.ke_cutoff = cell0.ke_cutoff
    cell.verbose = 0
    cell.build()

    logger.info(cell0, "generating init guess using %s basis", cell.basis)

    if len(kpts) < 30:
        pmf = scf.KRHF(cell, kpts)
    else:
        pmf = scf.KRHF(cell, kpts).density_fit()

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
    elif key.lower() == "scf":
        pmf.kernel()
        mo_coeff = pmf.mo_coeff
        mo_occ = pmf.mo_occ
    else:
        raise NotImplementedError("Init guess %s not implemented" % key)

    # TODO: support specifying nvir for each kpt (useful for e.g., metals)
    assert(isinstance(nvir, int) and nvir >= 0)
    nocc = cell.nelectron // 2
    nmo_ks = [len(mo_occ[k]) for k in range(nkpts)]
    ntot = nocc + nvir
    ntot_ks = [min(ntot,nmo_ks[k]) for k in range(nkpts)]

    logger.debug1(cell0, "converting init MOs from GTO basis to PW basis")
    C_ks = pw_helper.get_C_ks_G(cell, kpts, mo_coeff, ntot_ks, fC_ks=fC_ks,
                                verbose=cell0.verbose)
    mocc_ks = [mo_occ[k][:ntot_ks[k]] for k in range(nkpts)]

    C_ks = orth_mo(cell0, C_ks, mocc_ks)

    C_ks, mocc_ks = add_random_mo(cell0, [ntot]*nkpts, C_ks, mocc_ks)

    return C_ks, mocc_ks


def add_random_mo(cell, n_ks, C_ks, mocc_ks):
    """ Add random MOs if C_ks[k].shape[0] < n_ks[k] for any k
    """
    nkpts = len(n_ks)
    incore = isinstance(C_ks, list)
    for k in range(nkpts):
        n = n_ks[k]
        n0 = C_ks[k].shape[0] if incore else C_ks["%d"%k].shape[0]
        if n0 < n:
            n1 = n - n0
            logger.warn(cell, "Requesting more orbitals than currently have (%d > %d) for kpt %d. Adding %d random orbitals.", n, n0, k, n1)
            C0 = C_ks[k] if incore else C_ks["%d"%k][()]
            C = add_random_mo1(cell, n, C0)
            if incore:
                C_ks[k] = C
            else:
                del C_ks["%d"%k]
                C_ks["%d"%k] = C

            mocc = mocc_ks[k]
            mocc_ks[k] = np.concatenate([mocc, np.zeros(n1,dtype=mocc.dtype)])

    return C_ks, mocc_ks


def add_random_mo1(cell, n, C0):
    n0, ngrids = C0.shape
    if n == n0:
        return C0

    C1 = np.random.rand(n-n0, ngrids) + 0j
    C1 -= lib.dot(lib.dot(C1, C0.conj().T), C0)
    C1 = pw_helper.orth(cell, C1, 1e-3, follow=False)

    return np.vstack([C0,C1])


def init_guess_by_chkfile(cell, chkfile_name, nvir, project=None):
    from pyscf.pbc.scf import chkfile
    scf_dict = chkfile.load_scf(chkfile_name)[1]
    mocc_ks = scf_dict["mo_occ"]
    nkpts = len(mocc_ks)
    ntot_ks = [None] * nkpts
    for k in range(nkpts):
        nocc = np.sum(mocc_ks[k]>THR_OCC)
        ntot_ks[k] = max(nocc+nvir, len(mocc_ks[k]))

    fchk = h5py.File(chkfile_name, "a")
    C_ks = fchk["mo_coeff"]

    C_ks, mocc_ks = init_guess_from_C0(cell, C_ks, ntot_ks, C_ks, mocc_ks)

    return fchk, C_ks, mocc_ks


def init_guess_from_C0(cell, C0_ks, ntot_ks, C_ks=None, mocc_ks=None):
    nkpts = len(C0_ks)
    if C_ks is None: C_ks = [None] * nkpts
    incore0 = isinstance(C0_ks, list)
    incore = isinstance(C_ks, list)

    # discarded high-energy orbitals if chkfile has more than requested
    for k in range(nkpts):
        key = "%d"%k
        ntot = ntot_ks[k]
        C0_k = C0_ks[k] if incore0 else C0_ks[key][()]
        if C0_k.shape[0] > ntot:
            C = C0_k[:ntot]
            if not mocc_ks is None:
                mocc_ks[k] = mocc_ks[k][:ntot]
        else:
            C = C0_k
        if incore:
            C_ks[k] = C
        else:
            if key in C_ks: del C_ks[key]
            C_ks[key] = C

    if mocc_ks is None:
        mocc_ks = get_mo_occ(cell, C_ks=C_ks)

    C_ks = orth_mo(cell, C_ks, mocc_ks)

    C_ks, mocc_ks = add_random_mo(cell, ntot_ks, C_ks, mocc_ks)

    return C_ks, mocc_ks


def update_subspace_vppnloc(mf, pp, C_ks):
    pp.update_subspace_vppnloc(C_ks)


def initialize_ACE(mf, C_ks, mocc_ks, kpts, mesh, Gv, ace_xi_ks, Ct_ks=None):
    tick = np.asarray([time.clock(), time.time()])
    if not "t-ace" in mf.scf_summary:
        mf.scf_summary["t-ace"] = np.zeros(2)

    if not ace_xi_ks is None:
        cell = mf.cell
        pw_helper.initialize_ACE(cell, C_ks, mocc_ks, kpts, mesh, Gv,
                                 ace_xi_ks, Ct_ks=Ct_ks)

    tock = np.asarray([time.clock(), time.time()])
    mf.scf_summary["t-ace"] += tock - tick


def apply_h1e_kpt(mf, C_k, kpt, mesh, Gv, pp, ret_E=False):
    r""" Apply 1e part of the Fock opeartor to orbitals at given k-point. The local part includes kinetic and pseudopotential (both local and non-local).
    """
    res = apply_Fock_local_kpt(mf.cell, C_k, kpt, mesh, Gv, pp, None,
                               ret_E=ret_E)

    Cbar_k = res[0]

    if ret_E:
        es = res[1][:3]
        if (np.abs(es.imag) > 1e-6).any():
            e_comp = mf.scf_summary["e_comp_name_lst"]
            icomps = np.where(np.abs(es.imag) > 1e-6)[0]
            logger.warn(mf, "Energy has large imaginary part:" +
                     "%s : %s\n" * len(icomps),
                     *[s for i in icomps for s in [e_comp[i],es[i]]])
        es = es.real

    tspans = res[-1][:3]
    for icomp,comp in enumerate(mf.scf_summary["e_comp_name_lst"][:3]):
        key = "t-%s" % comp
        if not key in mf.scf_summary:
            mf.scf_summary[key] = np.zeros(2)
        mf.scf_summary[key] += tspans[icomp]

    if ret_E:
        return Cbar_k, es
    else:
        return Cbar_k


def apply_Fock_local_kpt(cell, C_k, kpt, mesh, Gv, pp, vj_R, ret_E=False):
    r""" Apply local part of the Fock opeartor to orbitals at given k-point. The local part includes kinetic, pseudopotential (both local and non-local), and Hartree.
    """
    es = np.zeros(4, dtype=np.complex128)

    tspans = np.zeros((4,2))
    tick = np.asarray([time.clock(), time.time()])

    tmp = pw_helper.apply_kin_kpt(C_k, kpt, mesh, Gv)
    Cbar_k = tmp
    es[0] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2
    tock = np.asarray([time.clock(), time.time()])
    tspans[0] = tock - tick

    C_k_R = tools.ifft(C_k, mesh)
    tmp = tools.fft(C_k_R * pp.vpplocR, mesh)
    Cbar_k += tmp
    es[1] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2
    tick = np.asarray([time.clock(), time.time()])
    tspans[1] = tick - tock

    tmp = pp.apply_vppnl_kpt(C_k, kpt, mesh=mesh, Gv=Gv)
    Cbar_k += tmp
    es[2] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2
    tock = np.asarray([time.clock(), time.time()])
    tspans[2] = tock - tick

    if not vj_R is None:
        tmp = tools.fft(C_k_R * vj_R, mesh)
        Cbar_k += tmp * 2
        es[3] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2.
        tick = np.asarray([time.clock(), time.time()])
        tspans[3] = tick - tock

    C_k_R = None

    if ret_E:
        return Cbar_k, es, tspans
    else:
        return Cbar_k, tspans


def apply_Fock_nonlocal_kpt(cell, C_k, kpt, mocc_ks, kpts, mesh, Gv, exxdiv,
                            C_ks_exx=None, ace_xi_k=None,
                            ret_E=False):
    r""" Apply non-local part of the Fock opeartor to orbitals at given k-point. The non-local part includes the exact exchange.
    """
    tick = np.asarray([time.clock(), time.time()])
    if ace_xi_k is None:
        assert(not C_ks_exx is None)
        Cbar_k = pw_helper.apply_vk_kpt(cell, C_k, kpt, C_ks_exx, mocc_ks, kpts,
                                        mesh=mesh, Gv=Gv)
    else:
        Cbar_k = pw_helper.apply_vk_kpt_ace(C_k, ace_xi_k)
    Cbar_k *= -1
    e = np.einsum("ig,ig->", C_k.conj(), Cbar_k)
    tock = np.asarray([time.clock(), time.time()])
    tspans = np.asarray(tock - tick).reshape(1,2)

    if ret_E:
        return Cbar_k, e, tspans
    else:
        return Cbar_k, tspans


def apply_Fock_kpt(mf, C_k, kpt, mocc_ks, mesh, Gv, pp, vj_R, exxdiv,
                   C_ks_exx=None, ace_xi_k=None, ret_E=False):
    """ Apply Fock operator to orbitals at given k-point.
    """
    cell = mf.cell
    kpts = mf.kpts
# local part
    res_l = apply_Fock_local_kpt(cell, C_k, kpt, mesh, Gv, pp, vj_R,
                                 ret_E=ret_E)
# nonlocal part
    res_nl = apply_Fock_nonlocal_kpt(cell, C_k, kpt, mocc_ks, kpts,
                                     mesh, Gv, exxdiv, C_ks_exx=C_ks_exx,
                                     ace_xi_k=ace_xi_k, ret_E=ret_E)
    Cbar_k = res_l[0] + res_nl[0]

    if ret_E:
        es = np.concatenate([res_l[1], [res_nl[1]]])
        if (np.abs(es.imag) > 1e-6).any():
            e_comp = mf.scf_summary["e_comp_name_lst"]
            icomps = np.where(np.abs(es.imag) > 1e-6)[0]
            logger.warn(mf, "Energy has large imaginary part:" +
                     "%s : %s\n" * len(icomps),
                     *[s for i in icomps for s in [e_comp[i],es[i]]])
        es = es.real

    tspans = np.vstack([res_l[-1], res_nl[-1]])
    for icomp,comp in enumerate(mf.scf_summary["e_comp_name_lst"]):
        key = "t-%s" % comp
        if not key in mf.scf_summary:
            mf.scf_summary[key] = np.zeros(2)
        mf.scf_summary[key] += tspans[icomp]

    if ret_E:
        return Cbar_k, es
    else:
        return Cbar_k


def get_mo_energy(mf, C_ks, mocc_ks, mesh=None, Gv=None, exxdiv=None,
                  C_ks_exx=None, ace_xi_ks=None, pp=None, vj_R=None,
                  ret_mocc=True):
    cell = mf.cell
    if pp is None: pp = pw_pseudo.PWPP(cell, mf.kpts, mesh=mesh)
    if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    if exxdiv is None: exxdiv = mf.exxdiv

    C_incore = isinstance(C_ks, list)

    kpts = mf.kpts
    nkpts = len(kpts)
    moe_ks = [None] * nkpts
    for k in range(nkpts):
        kpt = kpts[k]
        C_k = C_ks[k] if C_incore else C_ks["%d"%k][()]
        ace_xi_k = None if ace_xi_ks is None else ace_xi_ks["%d"%k][()]
        Cbar_k = mf.apply_Fock_kpt(C_k, kpt, mocc_ks, mesh, Gv, pp, vj_R,
                                   exxdiv, C_ks_exx=C_ks_exx,
                                   ace_xi_k=ace_xi_k, ret_E=False)
        moe_k = np.einsum("ig,ig->i", C_k.conj(), Cbar_k)
        if (moe_k.imag > 1e-6).any():
            logger.warn(mf, "MO energies have imaginary part %s for kpt %d",
                        moe_k, k)
        moe_ks[k] = moe_k.real

    if ret_mocc:
        mocc_ks = mf.get_mo_occ(moe_ks)
        if exxdiv == "ewald":
            for k in range(nkpts):
                moe_ks[k][mocc_ks[k] > THR_OCC] -= mf._madelung

        return moe_ks, mocc_ks
    else:
        return moe_ks


def energy_elec(mf, C_ks, mocc_ks, mesh=None, Gv=None, moe_ks=None,
                C_ks_exx=None, ace_xi_ks=None, pp=None, vj_R=None,
                exxdiv=None):
    ''' Compute the electronic energy
    Pass `moe_ks` to avoid the cost of applying the expensive vj and vk.
    '''
    cell = mf.cell
    if pp is None: pp = pw_pseudo.PWPP(cell, mf.kpts, mesh=mesh)
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)
    if exxdiv is None: exxdiv = mf.exxdiv

    C_incore = isinstance(C_ks, list)

    kpts = mf.kpts
    nkpts = len(kpts)

    nocc_ks = [np.sum(mocc_ks[k] > 0) for k in range(nkpts)]

    e_ks = np.zeros(nkpts)
    if moe_ks is None:
        if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)
        e_comp = np.zeros(5)
        for k in range(nkpts):
            kpt = kpts[k]
            nocc_k = nocc_ks[k]
            Co_k = C_ks[k][:nocc_k] if C_incore else C_ks["%d"%k][:nocc_k]
            ace_xi_k = None if ace_xi_ks is None else ace_xi_ks["%d"%k][()]
            e_comp_k = mf.apply_Fock_kpt(Co_k, kpt, mocc_ks, mesh, Gv,
                                         pp, vj_R, exxdiv,
                                         C_ks_exx=C_ks_exx, ace_xi_k=ace_xi_k,
                                         ret_E=True)[1]
            e_ks[k] = np.sum(e_comp_k)
            e_comp += e_comp_k
        e_comp /= nkpts

        if exxdiv == "ewald":
            e_comp[mf.scf_summary["e_comp_name_lst"].index("ex")] += \
                                                        mf._etot_shift_ewald

        for comp,e in zip(mf.scf_summary["e_comp_name_lst"],e_comp):
            mf.scf_summary[comp] = e
    else:
        for k in range(nkpts):
            kpt = kpts[k]
            nocc_k = nocc_ks[k]
            Co_k = C_ks[k][:nocc_k] if C_incore else C_ks["%d"%k][:nocc_k]
            e1_comp = mf.apply_h1e_kpt(Co_k, kpt, mesh, Gv,
                                       pp, ret_E=True)[1]
            e_ks[k] = np.sum(e1_comp) * 0.5 + np.sum(moe_ks[k][:nocc_k])
    e_scf = np.sum(e_ks) / nkpts

    if moe_ks is None and exxdiv == "ewald":
        # Note: ewald correction is not needed if e_tot is computed from moe_ks since the correction is already in the mo energy
        e_scf += mf._etot_shift_ewald

    return e_scf


def energy_tot(mf, C_ks, mocc_ks, moe_ks=None, mesh=None, Gv=None,
               C_ks_exx=None, ace_xi_ks=None,
               pp=None, vj_R=None, exxdiv=None):
    e_nuc = mf.scf_summary["nuc"]
    e_scf = mf.energy_elec(C_ks, mocc_ks, moe_ks=moe_ks, mesh=mesh, Gv=Gv,
                           C_ks_exx=C_ks_exx, ace_xi_ks=ace_xi_ks,
                           pp=pp, vj_R=vj_R, exxdiv=exxdiv)
    e_tot = e_scf + e_nuc
    return e_tot


def converge_band_kpt(mf, C_k, kpt, mocc_ks, nband=None, mesh=None, Gv=None,
                      C_ks_exx=None, ace_xi_k=None,
                      pp=None, vj_R=None,
                      conv_tol_davidson=1e-6,
                      max_cycle_davidson=100,
                      verbose_davidson=0):
    ''' Converge all occupied orbitals for a given k-point using davidson algorithm
    '''

    fc = [0]
    def FC(C_k_, ret_E=False):
        fc[0] += 1
        C_k_ = np.asarray(C_k_)
        Cbar_k_ = mf.apply_Fock_kpt(C_k_, kpt, mocc_ks, mesh, Gv,
                                    pp, vj_R, "none",
                                    C_ks_exx=C_ks_exx, ace_xi_k=ace_xi_k,
                                    ret_E=False)
        return Cbar_k_

    tick = np.asarray([time.clock(), time.time()])

    kG = kpt + Gv if np.sum(np.abs(kpt)) > 1.E-9 else Gv
    dF = np.einsum("gj,gj->g", kG, kG) * 0.5
    precond = lambda dx, e, x0: dx/(dF - e)

    nroots = C_k.shape[0] if nband is None else nband

    conv, e, c = lib.davidson1(FC, C_k, precond,
                               nroots=nroots,
                               verbose=verbose_davidson,
                               tol=conv_tol_davidson,
                               max_cycle=max_cycle_davidson)
    c = np.asarray(c)

    tock = np.asarray([time.clock(), time.time()])
    key = "t-dvds"
    if not key in mf.scf_summary:
        mf.scf_summary[key] = np.zeros(2)
    mf.scf_summary[key] += tock - tick

    return conv, e, c, fc[0]


def converge_band(mf, C_ks, mocc_ks, kpts, Cout_ks=None,
                  mesh=None, Gv=None,
                  C_ks_exx=None, ace_xi_ks=None,
                  pp=None, vj_R=None,
                  conv_tol_davidson=1e-6,
                  max_cycle_davidson=100,
                  verbose_davidson=0):
    if pp is None: pp = pw_pseudo.PWPP(cell, mf.kpts, mesh=mesh)
    if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)

    C_incore = isinstance(C_ks, list)

    nkpts = len(kpts)
    if C_incore:
        if Cout_ks is None: Cout_ks = [None] * nkpts
    else:
        Cout_ks = C_ks
    conv_ks = [None] * nkpts
    moeout_ks = [None] * nkpts
    fc_ks = [None] * nkpts

    for k in range(nkpts):
        kpt = kpts[k]
        C_k = C_ks[k] if C_incore else C_ks["%d"%k][()]
        ace_xi_k = None if ace_xi_ks is None else ace_xi_ks["%d"%k][()]
        conv_, moeout_ks[k], Cout_k, fc_ks[k] = \
                    mf.converge_band_kpt(C_k, kpt, mocc_ks,
                                         mesh=mesh, Gv=Gv,
                                         C_ks_exx=C_ks_exx,
                                         ace_xi_k=ace_xi_k,
                                         pp=pp, vj_R=vj_R,
                                         conv_tol_davidson=conv_tol_davidson,
                                         max_cycle_davidson=max_cycle_davidson,
                                         verbose_davidson=verbose_davidson)
        if C_incore:
            Cout_ks[k] = Cout_k
        else:
            Cout_ks["%d"%k][()] = Cout_k
        conv_ks[k] = np.prod(conv_)

    return conv_ks, moeout_ks, Cout_ks, fc_ks


# class PWKRHF(lib.StreamObject):
class PWKRHF(mol_hf.SCF):
    '''PWKRHF base class. non-relativistic RHF using PW basis.
    '''

    conv_tol = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_conv_tol', 1e-6)
    conv_tol_davidson = getattr(__config__,
                                'pbc_pwscf_khf_PWKRHF_conv_tol_davidson', 1e-7)
    conv_tol_band = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_conv_tol_band',
                            1e-4)
    max_cycle = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_max_cycle', 100)
    max_cycle_davidson = getattr(__config__,
                                 'pbc_pwscf_khf_PWKRHF_max_cycle_davidson',
                                 100)
    verbose_davidson = getattr(__config__,
                               'pbc_pwscf_khf_PWKRHF_verbose_davidson', 0)
    double_loop_scf = getattr(__config__,
                              'pbc_pwscf_khf_PWKRHF_double_loop_scf', True)
    ace_exx = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_ace_exx', True)
    damp_type = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_damp_type',
                        "anderson")
    damp_factor = getattr(__config__, 'pbc_pwscf_khf_PWKRHF_damp_factor', 0.3)
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
            self._madelung = tools.pbc.madelung(self.cell, self.kpts)
            self._etot_shift_ewald = -0.5*self._madelung*cell.nelectron
        self.scf_summary["nuc"] = self.cell.energy_nuc()
        self.scf_summary["e_comp_name_lst"] = ["kin", "ppl", "ppnl", "coul", "ex"]

        self.nvir = 0 # number of virtual bands to compute
        self.nvir_extra = 1 # to facilitate converging the highest virtual
        self.init_guess = "hcore"

# If _acexi_to_save is specified (as a str), the ACE xi vectors will be saved in this file. Otherwise, a tempfile is used and discarded after the calculation.
        self._acexi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)

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
        logger.info(self, "Num virtual bands to compute = %s", self.nvir)
        logger.info(self, "Num extra v-bands included to help convergence = %s",
                    self.nvir_extra)
        logger.info(self, "Band energy conv_tol = %s", self.conv_tol_band)
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
        if isinstance(self._acexi_to_save, str):
            logger.info(self, '_acexi_to_save = %s', self._acexi_to_save)
        else:
            logger.info(self, '_acexi_to_save = %s', self._acexi_to_save.name)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])

        logger.info(self, 'kpts = %s', self.kpts)
        logger.info(self, 'Exchange divergence treatment (exxdiv) = %s',
                    self.exxdiv)

        cell = self.cell
        if ((cell.dimension >= 2 and cell.low_dim_ft_type != 'inf_vacuum') and
            isinstance(self.exxdiv, str) and self.exxdiv.lower() == 'ewald'):
            madelung = self._madelung
            logger.info(self, '    madelung (= occupied orbital energy shift) = %s', madelung)
            logger.info(self, '    Total energy shift due to Ewald probe charge'
                        ' = -1/2 * Nelec*madelung = %.12g',
                        madelung*cell.nelectron * -.5)

    def init_guess_by_chkfile(self, chk=None, nvir=None, project=None):
        if chk is None: chk = self.chkfile
        if nvir is None: nvir = self.nvir
        return init_guess_by_chkfile(self.cell, chk, nvir, project=project)
    def from_chk(self, chk=None, project=None, kpts=None):
        return self.init_guess_by_chkfile(chk, project, kpts)

    def dump_chk(self, envs):
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile,
                             envs['e_tot'], envs['moe_ks'],
                             envs['mocc_ks'], envs['C_ks'])
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

        def write_time(comp, t_comp, t_tot):
            tc, tw = t_comp
            tct, twt = t_tot
            rc = tc / tct * 100
            rw = tw / twt * 100
            log.info('CPU time for %10s %9.2f  ( %6.2f%% ), wall time %9.2f  ( %6.2f%% )', comp.ljust(10), tc, rc, tw, rw)

        t_tot = summary["t-tot"]
        write_time("init guess", summary["t-init"], t_tot)
        write_time("init ACE", summary["t-ace"], t_tot)
        t_fock = np.zeros(2)
        for op in summary["e_comp_name_lst"]:
            write_time("op %s"%op, summary["t-%s"%op], t_tot)
            t_fock += summary["t-%s"%op]
        t_dvds = np.clip(summary['t-dvds']-t_fock, 0, None)
        write_time("dvds other", t_dvds, t_tot)
        t_other = t_tot - summary["t-init"] - summary["t-ace"] - \
                    summary["t-dvds"]
        write_time("all other", t_other, t_tot)
        write_time("full SCF", t_tot, t_tot)

    def get_mo_occ(mf, moe_ks=None, C_ks=None, nocc=None):
        return get_mo_occ(mf.cell, moe_ks, C_ks, nocc)

    def get_init_guess_key(self, cell=None, kpts=None, basis=None, pseudo=None,
                           nvir=None, key="hcore", fC_ks=None):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if nvir is None: nvir = self.nvir

        if key in ["h1e","hcore","cycle1","scf"]:
            C_ks, mocc_ks = get_init_guess(cell, kpts,
                                           basis=basis, pseudo=pseudo,
                                           nvir=nvir, key=key, fC_ks=fC_ks)
        else:
            logger.warn(self, "Unknown init guess %s", key)
            raise RuntimeError

        return C_ks, mocc_ks

    def get_init_guess(self, init_guess=None, nvir=None, chkfile=None, C0=None):
        if init_guess is None: init_guess = self.init_guess
        if nvir is None: nvir = self.nvir
        if chkfile is None: chkfile = self.chkfile
        if init_guess[:3] == "chk" and C0 is None:
            fchk, C_ks, mocc_ks = self.init_guess_by_chkfile(chk=chkfile, nvir=nvir)
            dump_chk = True
        else:
            if isinstance(chkfile, str):
                fchk = h5py.File(chkfile, "w")
                dump_chk = True
            else:
                # tempfile (discarded after SCF)
                swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
                fchk = lib.H5TmpFile(swapfile.name)
                swapfile = None
                dump_chk = False

            C_ks = fchk.create_group("mo_coeff")

            if C0 is None:
                C_ks, mocc_ks = self.get_init_guess_key(nvir=nvir,
                                                        key=init_guess,
                                                        fC_ks=C_ks)
            else:
                C_ks, mocc_ks = self.get_init_guess_C0(C0, nvir=nvir, fC_ks=C_ks)

        return C_ks, mocc_ks, fchk, dump_chk

    def get_init_guess_C0(self, C0, nvir=None, fC_ks=None):
        if nvir is None: nvir = self.nvir
        nocc = self.cell.nelectron // 2
        ntot_ks = [nocc+nvir] * len(self.kpts)
        return init_guess_from_C0(self.cell, C0, ntot_ks, fC_ks)

    def get_vj_R(self, C_ks, mocc_ks, mesh=None, Gv=None):
        return pw_helper.get_vj_R(self.cell, C_ks, mocc_ks, mesh=mesh, Gv=Gv)

    def scf(self, C0=None, **kwargs):
        self.dump_flags()

        if isinstance(self._acexi_to_save, tempfile._TemporaryFileWrapper):
            facexi = lib.H5TmpFile(self._acexi_to_save.name)
        else:
            facexi = self._acexi_to_save
        self.converged, self.e_tot, self.mo_energy, self.mo_coeff, \
                self.mo_occ = kernel_doubleloop(self, self.kpts,
                           C0=C0, facexi=facexi,
                           nbandv=self.nvir, nbandv_extra=self.nvir_extra,
                           conv_tol=self.conv_tol, max_cycle=self.max_cycle,
                           conv_tol_band=self.conv_tol_band,
                           conv_tol_davidson=self.conv_tol_davidson,
                           max_cycle_davidson=self.max_cycle_davidson,
                           verbose_davidson=self.verbose_davidson,
                           ace_exx=self.ace_exx,
                           damp_type=self.damp_type,
                           damp_factor=self.damp_factor,
                           conv_check=self.conv_check,
                           callback=self.callback,
                           **kwargs)
        self._finalize()
        return self.e_tot
    kernel = lib.alias(scf, alias_name='kernel')

    kernel_charge = kernel_charge
    remove_extra_virbands = remove_extra_virbands
    get_nband = get_nband
    get_band_err = get_band_err
    copy_C_ks = copy_C_ks
    dump_moe = dump_moe
    update_subspace_vppnloc = update_subspace_vppnloc
    initialize_ACE = initialize_ACE
    get_mo_energy = get_mo_energy
    apply_h1e_kpt = apply_h1e_kpt
    apply_Fock_kpt = apply_Fock_kpt
    energy_elec = energy_elec
    energy_tot = energy_tot
    converge_band_kpt = converge_band_kpt
    converge_band = converge_band


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
    mf.nvir = 4 # converge first 4 virtual bands
    mf.kernel()
    assert(abs(mf.e_tot - -11.0411939355984) < 1.e-5)

    mf.dump_scf_summary()
