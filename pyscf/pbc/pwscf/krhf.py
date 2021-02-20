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
from pyscf.pbc.lib.kpts_helper import member
from pyscf.lib import logger
import pyscf.lib.parameters as param


THR_OCC = 1E-3


def kernel_doubleloop(mf, kpts, C0_ks=None, facexi=None,
            nbandv=0, nbandv_extra=1,
            conv_tol=1.E-6, conv_tol_davidson=1.E-6, conv_tol_band=1e-4,
            max_cycle=100, max_cycle_davidson=10, verbose_davidson=0,
            ace_exx=True, damp_type="anderson", damp_factor=0.3,
            conv_check=True, callback=None, **kwargs):
    ''' Kernel function for SCF in a PW basis
        Note:
            This double-loop implementation follows closely the implementation in Quantum ESPRESSO.

        Args:
            C0_ks (list of numpy arrays):
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

    nbando = cell.nelectron // 2
    nbandv_tot = nbandv + nbandv_extra
    nband = nbando + nbandv
    nband_tot = nbando + nbandv_tot
    logger.info(mf, "Num of occ bands= %d", nbando)
    logger.info(mf, "Num of vir bands= %d", nbandv)
    logger.info(mf, "Num of all bands= %d", nband)
    logger.info(mf, "Num of extra vir bands= %d", nbandv_extra)

    # init guess and SCF chkfile
    tick = np.asarray([time.clock(), time.time()])
    if mf.init_guess[:3] == "chk":
        fchk, C_ks, mocc_ks = mf.init_guess_by_chkfile(nv=nbandv_tot)
        dump_chk = True
    else:
        if isinstance(mf.chkfile, str):
            fchk = h5py.File(mf.chkfile, "w")
            dump_chk = True
        else:
            # tempfile (discarded after SCF)
            swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            fchk = lib.H5TmpFile(swapfile.name)
            swapfile = None
            dump_chk = False
        C_ks = fchk.create_group("mo_coeff")

        if C0_ks is None:
            C_ks, mocc_ks = mf.get_init_guess(nv=nbandv_tot, key=mf.init_guess,
                                              fC_ks=C_ks)
        else:
            C_ks = C0_ks
            mocc_ks = get_mo_occ(cell, C_ks=C_ks)
            C_ks, mocc_ks = add_random_mo(mf.cell, [nband_tot]*nkpts, C_ks,
                                          mocc_ks)
    tock = np.asarray([time.clock(), time.time()])
    mf.scf_summary["t-init"] = tock - tick

    # file for store acexi
    if ace_exx:
        if facexi is None:
            swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            logger.info(mf, "ACE xi vectors will be written to %s", swapfile.name)
            facexi = lib.H5TmpFile(swapfile.name)
            swapfile = None
        elif isinstance(facexi, str):
            facexi = h5py.File(facexi, "w")
    else:
        facexi = None

    # init E
    mesh = cell.mesh
    Gv = cell.get_Gv(mesh)
    vpplocR = mf.get_vpplocR(mesh=mesh, Gv=Gv)
    vj_R = mf.get_vj_R(C_ks, mocc_ks, mesh=mesh, Gv=Gv)
    mf.initialize_ACE(C_ks, mocc_ks, kpts, mesh, Gv, facexi=facexi)
    C_ks_exx = list(C_ks) if facexi is None else None
    moe_ks = mf.get_mo_energy(C_ks, mocc_ks, C_ks_exx=C_ks_exx, facexi=facexi,
                              vpplocR=vpplocR, vj_R=vj_R)
    # mocc_ks = get_mo_occ(cell, moe_ks)
    # sort_mo(mocc_ks, moe_ks, C_ks)
    e_tot = mf.energy_tot(C_ks, mocc_ks, moe_ks=moe_ks, vpplocR=vpplocR)
    logger.info(mf, 'init E= %.15g', e_tot)
    mf.dump_moe(moe_ks, mocc_ks, nband=nband)

    scf_conv = False

    if mf.max_cycle <= 0:
        remove_extra_virbands(C_ks, moe_ks, mocc_ks, nbandv_extra)
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
        chg_scf_conv, fc_this, C_ks, chg_moe_ks, chg_e_tot = mf.kernel_charge(
                                C_ks, mocc_ks, kpts, nband, mesh=mesh, Gv=Gv,
                                max_cycle=max_cycle, conv_tol=chg_conv_tol,
                                max_cycle_davidson=max_cycle_davidson,
                                conv_tol_davidson=conv_tol_davidson,
                                verbose_davidson=verbose_davidson,
                                damp_type=damp_type, damp_factor=damp_factor,
                                C_ks_exx=C_ks_exx, facexi=facexi,
                                vpplocR=vpplocR, vj_R=vj_R,
                                last_hf_e=e_tot)
        fc_tot += fc_this
        if not chg_scf_conv:
            logger.warn(mf, "  Charge SCF not converged.")

        # update coulomb potential, facexi, and energies
        vj_R = mf.get_vj_R(C_ks, mocc_ks)
        mf.initialize_ACE(C_ks, mocc_ks, kpts, mesh, Gv, facexi=facexi)
        C_ks_exx = list(C_ks) if facexi is None else None
        last_hf_moe = moe_ks
        moe_ks = mf.get_mo_energy(C_ks, mocc_ks,
                                  C_ks_exx=C_ks_exx, facexi=facexi,
                                  vpplocR=vpplocR, vj_R=vj_R)
        de_band = np.max([np.max(abs(moe_ks[k] - last_hf_moe[k])[:nband])
                         for k in range(nkpts)])
        mocc_ks = get_mo_occ(cell, moe_ks)
        last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, mocc_ks, C_ks_exx=C_ks_exx, facexi=facexi,
                              vpplocR=vpplocR, vj_R=vj_R)
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

        chg_scf_conv, fc_this, C_ks, chg_moe_ks, chg_e_tot = mf.kernel_charge(
                                C_ks, mocc_ks, kpts, nband, mesh=mesh, Gv=Gv,
                                max_cycle=max_cycle, conv_tol=chg_conv_tol,
                                max_cycle_davidson=max_cycle_davidson,
                                conv_tol_davidson=conv_tol_davidson,
                                verbose_davidson=verbose_davidson,
                                damp_type=damp_type, damp_factor=damp_factor,
                                C_ks_exx=C_ks_exx, facexi=facexi,
                                vpplocR=vpplocR, vj_R=vj_R,
                                last_hf_e=e_tot)
        fc_tot += fc_this
        vj_R = mf.get_vj_R(C_ks, mocc_ks)
        mf.initialize_ACE(C_ks, mocc_ks, kpts, mesh, Gv, facexi=facexi)
        C_ks_exx = list(C_ks) if facexi is None else None
        last_hf_moe = moe_ks
        moe_ks = mf.get_mo_energy(C_ks, mocc_ks,
                                  C_ks_exx=C_ks_exx, facexi=facexi,
                                  vpplocR=vpplocR, vj_R=vj_R)
        de_band = np.max([np.max(abs(moe_ks[k] - last_hf_moe[k])[:nband])
                         for k in range(nkpts)])
        mocc_ks = get_mo_occ(cell, moe_ks)
        last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, mocc_ks, C_ks_exx=C_ks_exx, facexi=facexi,
                              vpplocR=vpplocR, vj_R=vj_R)
        de = e_tot - last_hf_e

        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  max|dEband|= %4.3g  %d FC (%d tot)',
                    e_tot, de, de_band, fc_this, fc_tot)
        mf.dump_moe(moe_ks, mocc_ks, nband=nband)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(de) < conv_tol and abs(de_band) < conv_tol_band:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    # remove extra virtual bands before return
    remove_extra_virbands(C_ks, moe_ks, mocc_ks, nbandv_extra)

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
#         no = np.sum(mocc>THR_OCC)
#         if np.sum(mocc[:no]>THR_OCC) < no:
#             mocc_ks[k] = np.asarray([2 if i < no else 0
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


def remove_extra_virbands(C_ks, moe_ks, mocc_ks, nbandv_extra):
    if nbandv_extra > 0:
        nkpts = len(moe_ks)
        moe_ks = [moe_ks[k][:-nbandv_extra] for k in range(nkpts)]
        mocc_ks = [mocc_ks[k][:-nbandv_extra] for k in range(nkpts)]
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
                  C_ks_exx=None, facexi=None,
                  vpplocR=None, vj_R=None,
                  last_hf_e=None):

    cell = mf.cell
    if vpplocR is None: vpplocR = mf.get_vpplocR()
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
                            C_ks_exx=C_ks_exx, facexi=facexi,
                            vpplocR=vpplocR, vj_R=vj_R,
                            conv_tol_davidson=conv_tol_davidson,
                            max_cycle_davidson=max_cycle_davidson,
                            verbose_davidson=verbose_davidson)
        fc_this = sum(fc_ks)
        fc_tot += fc_this

        # update coulomb potential and energy
        last_vj_R = vj_R
        vj_R = mf.get_vj_R(C_ks, mocc_ks)

        if cycle > 0: last_hf_e = e_tot
        e_tot = mf.energy_tot(C_ks, mocc_ks, C_ks_exx=C_ks_exx, facexi=facexi,
                              vpplocR=vpplocR, vj_R=vj_R)
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

    return scf_conv, fc_tot, C_ks, moe_ks, e_tot


def get_mo_occ(cell, moe_ks=None, C_ks=None):
    if not moe_ks is None:
        no = cell.nelectron // 2
        nkpts = len(moe_ks)
        no_tot = no * nkpts
        e_fermi = np.sort(np.concatenate(moe_ks))[no_tot-1]
        EPSILON = 1e-10
        mocc_ks = [None] * nkpts
        for k in range(nkpts):
            mocc_k = np.zeros(moe_ks[k].size)
            mocc_k[moe_ks[k] < e_fermi+EPSILON] = 2
            mocc_ks[k] = mocc_k
    elif not C_ks is None:
        no = cell.nelectron // 2
        if isinstance(C_ks, list):
            mocc_ks = [np.asarray([2 if i < no else 0
                       for i in range(C_k.shape[0])]) for C_k in C_ks]
        elif isinstance(C_ks, h5py.Group):
            mocc_ks = [np.asarray([2 if i < no else 0
                       for i in range(C_ks[k].shape[0])]) for k in C_ks]
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
        ehomo_ks = [np.max(moe_ks[k][mocc_ks[k]>THR_OCC])
                    for k in range(nkpts)]
        ehomo = np.max(ehomo_ks)
        khomos = np.where(abs(ehomo_ks-ehomo) < 1e-4)[0]
        logger.debug(mf, '  HOMO = %.15g  kpt'+' %d'*khomos.size,
                     ehomo, *khomos)
        if np.sum(mocc_ks[0]<THR_OCC) > 0:
            elumo_ks = [np.min(moe_ks[k][mocc_ks[k]<THR_OCC])
                        for k in range(nkpts)]
            elumo = np.min(elumo_ks)
            klumos = np.where(abs(elumo_ks-elumo) < 1e-4)[0]
            logger.debug(mf, '  LUMO = %.15g  kpt'+' %d'*klumos.size,
                         elumo, *klumos)
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


def orth(cell, C, thr, follow=True):
    n = C.shape[0]
    S = C.conj() @ C.T
    nonorth_err = np.max(np.abs(S - np.eye(S.shape[0])))
    if nonorth_err > thr:
        logger.warn(cell, "non-orthogonality detected in the initial MOs (max |off-diag ovlp|= %s). Symm-orth them now.", nonorth_err)
        e, u = scipy.linalg.eigh(S)
        if follow:
            # reorder to maximally overlap original orbs
            idx = []
            for i in range(n):
                order = np.argsort(np.abs(u[i]))[::-1]
                for j in order:
                    if not j in idx:
                        break
                idx.append(j)
            U = (u[:,idx]*e[idx]**-0.5).T
        else:
            U = (u*e**-0.5).T
        C = U @ C

    return C


def orth_mo1(cell, C, mocc, thr=1e-3):
    """ orth occupieds and virtuals separately
    """
    Co = C[mocc>THR_OCC]
    Cv = C[mocc<THR_OCC]
    # orth occ
    Co = orth(cell, Co, thr, follow=True)
    C[mocc>THR_OCC] = Co
    # project out occ from vir and orth vir
    if Cv.shape[0] > 0:
        Cv -= (Cv @ Co.conj().T) @ Co
        Cv = orth(cell, Cv, thr)
        C[mocc<THR_OCC] = Cv

    return C


def orth_mo(cell, C_ks, mocc_ks, thr=1e-3):
    nkpts = len(mocc_ks)
    if isinstance(C_ks, list):
        for k in range(nkpts):
            C_ks[k] = orth_mo1(cell, C_ks[k], mocc_ks[k], thr)
    elif isinstance(C_ks, h5py.Group):
        for k in range(nkpts):
            key = "%d"%k
            C_ks[key][()] = orth_mo1(cell, C_ks[key][()], mocc_ks[k], thr)
    else:
        raise RuntimeError

    return C_ks


def get_init_guess(mf, basis=None, pseudo=None, nv=0, key="hcore", fC_ks=None):
    """
        Args:
            nv (int):
                Number of virtual bands to be evaluated. Default is zero.
            fC_ks (h5py group):
                If provided, the orbitals are written to it.
    """

    if not fC_ks is None:
        assert(isinstance(fC_ks, h5py.Group))

    kpts = mf.kpts
    nkpts = len(kpts)

    if basis is None: basis = mf.cell.basis
    if pseudo is None: pseudo = mf.cell.pseudo
    cell = copy.copy(mf.cell)
    cell.basis = basis
    cell.pseudo = pseudo
    cell.ke_cutoff = mf.cell.ke_cutoff
    cell.verbose = 0
    cell.build()

    # TODO: support specifying nv for each kpt (useful for e.g., metals)
    assert(isinstance(nv, int) and nv >= 0)
    no = cell.nelectron // 2
    nao = cell.nao_nr()
    ntot = no + nv
    ntot_ks = [min(ntot,nao)] * nkpts

    logger.info(mf, "generating init guess using %s basis", cell.basis)

    if len(kpts) < 30:
        pmf = scf.KRHF(cell, kpts)
    else:
        pmf = scf.KRHF(cell, kpts).density_fit()
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

    logger.debug1(mf, "converting init MOs from GTO basis to PW basis")
    C_ks = pw_helper.get_C_ks_G(cell, kpts, mo_coeff, ntot_ks, fC_ks=fC_ks,
                                verbose=mf.cell.verbose)
    mocc_ks = get_mo_occ(cell, C_ks=C_ks)

    C_ks = orth_mo(mf, C_ks, mocc_ks)

    C_ks, mocc_ks = add_random_mo(mf.cell, [ntot]*nkpts, C_ks, mocc_ks)

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
    C1 -= (C1@C0.conj().T) @ C0
    C1 = orth(cell, C1, 1e-3, follow=False)

    return np.vstack([C0,C1])


def init_guess_by_chkfile(cell, chkfile_name, nv, project=None):
    fchk = h5py.File(chkfile_name, "a")
    C_ks = fchk["mo_coeff"]

    no = cell.nelectron // 2
    ntot = no + nv
    nkpts = len(C_ks)

    # discarded high-energy orbitals if chkfile has more than requested
    for k in range(nkpts):
        key = "%d"%k
        if C_ks[key].shape[0] > ntot:
            C = C_ks[key][:ntot]
            del C_ks[key]
            C_ks[key] = C
    mocc_ks = get_mo_occ(cell, C_ks=C_ks)

    C_ks = orth_mo(cell, C_ks, mocc_ks)

    C_ks, mocc_ks = add_random_mo(cell, [ntot]*nkpts, C_ks, mocc_ks)

    return fchk, C_ks, mocc_ks


def initialize_ACE(mf, C_ks, mocc_ks, kpts, mesh, Gv,
                   facexi=None, dataname="ace_xi", Ct_ks=None):
    tick = np.asarray([time.clock(), time.time()])
    if not "t-ace" in mf.scf_summary:
        mf.scf_summary["t-ace"] = np.zeros(2)

    if not facexi is None:
        cell = mf.cell
        pw_helper.initialize_ACE(cell, C_ks, mocc_ks, kpts, mesh, Gv,
                                 facexi, dataname, Ct_ks=Ct_ks)

    tock = np.asarray([time.clock(), time.time()])
    mf.scf_summary["t-ace"] += tock - tick


def apply_h1e_kpt(mf, C_k, kpt, mesh, Gv, vpplocR, ret_E=False):
    r""" Apply 1e part of the Fock opeartor to orbitals at given k-point. The local part includes kinetic and pseudopotential (both local and non-local).
    """
    res = apply_Fock_local_kpt(mf.cell, C_k, kpt, mesh, Gv, vpplocR, None,
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


def apply_Fock_local_kpt(cell, C_k, kpt, mesh, Gv, vpplocR, vj_R, ret_E=False):
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
    tmp = tools.fft(C_k_R * vpplocR, mesh)
    Cbar_k += tmp
    es[1] = np.einsum("ig,ig->", C_k.conj(), tmp) * 2
    tick = np.asarray([time.clock(), time.time()])
    tspans[1] = tick - tock

    tmp = pw_helper.apply_ppnl_kpt(cell, C_k, kpt, mesh, Gv)
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


def apply_Fock_nonlocal_kpt(cell, C_k, kpt, kpts, mesh, Gv, exxdiv, madelung,
                            C_ks_exx=None, mocc_ks=None, ace_xi_k=None,
                            ret_E=False):
    r""" Apply non-local part of the Fock opeartor to orbitals at given k-point. The non-local part includes the exact exchange.
    """
    tick = np.asarray([time.clock(), time.time()])
    if ace_xi_k is None:
        assert(not (C_ks_exx is None or mocc_ks is None))
        Cbar_k = pw_helper.apply_vk_kpt(cell, C_k, kpt, C_ks_exx, mocc_ks, kpts,
                                        mesh=mesh, Gv=Gv)
    else:
        Cbar_k = pw_helper.apply_vk_kpt_ace(C_k, ace_xi_k)
    if exxdiv == "ewald":
        k = member(kpt, kpts)[0]
        occ = mocc_ks[k] > 0
        Cbar_k[occ] += madelung * C_k[occ]
    elif exxdiv == "all":
        Cbar_k += madelung * C_k
    Cbar_k *= -1
    e = np.einsum("ig,ig->", C_k.conj(), Cbar_k)
    tock = np.asarray([time.clock(), time.time()])
    tspans = np.asarray(tock - tick).reshape(1,2)

    if ret_E:
        return Cbar_k, e, tspans
    else:
        return Cbar_k, tspans


def apply_Fock_kpt(mf, C_k, kpt, mesh, Gv, vpplocR, vj_R, exxdiv,
                   C_ks_exx=None, mocc_ks=None, ace_xi_k=None, ret_E=False):
    """ Apply Fock operator to orbitals at given k-point.
    """
    cell = mf.cell
    kpts = mf.kpts
# local part
    res_l = apply_Fock_local_kpt(cell, C_k, kpt, mesh, Gv, vpplocR, vj_R,
                                 ret_E=ret_E)
# nonlocal part
    madelung = mf.madelung
    res_nl = apply_Fock_nonlocal_kpt(cell, C_k, kpt, kpts,
                                     mesh, Gv, exxdiv, madelung,
                                     C_ks_exx=C_ks_exx, mocc_ks=mocc_ks,
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


def get_mo_energy(mf, C_ks, mocc_ks, mesh=None, Gv=None,
                  C_ks_exx=None, facexi=None, vpplocR=None, vj_R=None):
    cell = mf.cell
    if vpplocR is None: vpplocR = mf.get_vpplocR()
    if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)

    C_incore = isinstance(C_ks, list)

    kpts = mf.kpts
    nkpts = len(kpts)
    moe_ks = [None] * nkpts
    for k in range(nkpts):
        kpt = kpts[k]
        C_k = C_ks[k] if C_incore else C_ks["%d"%k][()]
        ace_xi_k = None if facexi is None else facexi["ace_xi/%d"%k][()]
        Cbar_k = mf.apply_Fock_kpt(C_k, kpt, mesh, Gv, vpplocR, vj_R, mf.exxdiv,
                                   C_ks_exx=C_ks_exx, mocc_ks=mocc_ks,
                                   ace_xi_k=ace_xi_k, ret_E=False)
        moe_k = np.einsum("ig,ig->i", C_k.conj(), Cbar_k)
        if (moe_k.imag > 1e-6).any():
            logger.warn(mf, "MO energies have imaginary part %s for kpt %d",
                        moe_k, k)
        moe_ks[k] = moe_k.real

    return moe_ks


def energy_elec(mf, C_ks, mocc_ks, mesh=None, Gv=None, moe_ks=None,
                C_ks_exx=None, facexi=None, vpplocR=None, vj_R=None,
                exxdiv=None):
    ''' Compute the electronic energy
    Pass `moe_ks` to avoid the cost of applying the expensive vj and vk.
    '''
    cell = mf.cell
    if vpplocR is None: vpplocR = mf.get_vpplocR()
    if mesh is None: mesh = cell.mesh
    if Gv is None: Gv = cell.get_Gv(mesh)

    C_incore = isinstance(C_ks, list)

    kpts = mf.kpts
    nkpts = len(kpts)

    no_ks = [np.sum(mocc_ks[k] > 0) for k in range(nkpts)]

    e_ks = np.zeros(nkpts)
    if moe_ks is None:
        if vj_R is None: vj_R = mf.get_vj_R(C_ks, mocc_ks)
        e_comp = np.zeros(5)
        for k in range(nkpts):
            kpt = kpts[k]
            no_k = no_ks[k]
            Co_k = C_ks[k][:no_k] if C_incore else C_ks["%d"%k][:no_k]
            ace_xi_k = None if facexi is None else facexi["ace_xi/%d"%k][()]
            e_comp_k = mf.apply_Fock_kpt(Co_k, kpt, mesh, Gv,
                                         vpplocR, vj_R, "all",
                                         C_ks_exx=C_ks_exx, mocc_ks=mocc_ks,
                                         ace_xi_k=ace_xi_k, ret_E=True)[1]
            e_ks[k] = np.sum(e_comp_k)
            e_comp += e_comp_k
        e_comp /= nkpts
        for comp,e in zip(mf.scf_summary["e_comp_name_lst"],e_comp):
            mf.scf_summary[comp] = e
    else:
        for k in range(nkpts):
            kpt = kpts[k]
            no_k = no_ks[k]
            Co_k = C_ks[k][:no_k] if C_incore else C_ks["%d"%k][:no_k]
            e1_comp = mf.apply_h1e_kpt(Co_k, kpt, mesh, Gv,
                                       vpplocR, ret_E=True)[1]
            e_ks[k] = np.sum(e1_comp) * 0.5 + np.sum(moe_ks[k][:no_k])
    e_scf = np.sum(e_ks) / nkpts

    return e_scf


def energy_tot(mf, C_ks, mocc_ks, moe_ks=None, mesh=None, Gv=None,
               C_ks_exx=None, facexi=None,
               vpplocR=None, vj_R=None, exxdiv=None):
    e_nuc = mf.scf_summary["nuc"]
    e_scf = mf.energy_elec(C_ks, mocc_ks, moe_ks=moe_ks, mesh=mesh, Gv=Gv,
                           C_ks_exx=C_ks_exx, facexi=facexi,
                           vpplocR=vpplocR, vj_R=vj_R, exxdiv=exxdiv)
    e_tot = e_scf + e_nuc
    return e_tot


def converge_band_kpt(mf, C_k, kpt, mocc_ks, nband=None, mesh=None, Gv=None,
                      C_ks_exx=None, ace_xi_k=None,
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
        Cbar_k_ = mf.apply_Fock_kpt(C_k_, kpt, mesh, Gv, vpplocR, vj_R, "all",
                                    C_ks_exx=C_ks_exx, mocc_ks=mocc_ks,
                                    ace_xi_k=ace_xi_k, ret_E=False)
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
                  C_ks_exx=None, facexi=None,
                  vpplocR=None, vj_R=None,
                  conv_tol_davidson=1e-6,
                  max_cycle_davidson=100,
                  verbose_davidson=0):
    if vpplocR is None: vpplocR = mf.get_vpplocR()
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
        ace_xi_k = None if facexi is None else facexi["ace_xi/%d"%k][()]
        conv_, moeout_ks[k], Cout_k, fc_ks[k] = \
                    mf.converge_band_kpt(C_k, kpt, mocc_ks,
                                         mesh=mesh, Gv=Gv,
                                         C_ks_exx=C_ks_exx,
                                         ace_xi_k=ace_xi_k,
                                         vpplocR=vpplocR, vj_R=vj_R,
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

    conv_tol = getattr(__config__, 'pbc_pwscf_krhf_PWKRHF_conv_tol', 1e-6)
    conv_tol_davidson = getattr(__config__,
                                'pbc_pwscf_krhf_PWKRHF_conv_tol_davidson', 1e-7)
    conv_tol_band = getattr(__config__, 'pbc_pwscf_krhf_PWKRHF_conv_tol_band',
                            1e-4)
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

        self.nv = 0 # number of virtual bands to compute
        self.nv_extra = 1    # to facilitate converging the highest virtual
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
        logger.info(self, "Num virtual bands to compute = %d", self.nv)
        logger.info(self, "Num extra v-bands included to help convergence = %d",
                    self.nv_extra)
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
            madelung = self.madelung
            logger.info(self, '    madelung (= occupied orbital energy shift) = %s', madelung)
            logger.info(self, '    Total energy shift due to Ewald probe charge'
                        ' = -1/2 * Nelec*madelung = %.12g',
                        madelung*cell.nelectron * -.5)

    def init_guess_by_chkfile(self, chk=None, nv=None, project=None):
        if chk is None: chk = self.chkfile
        if nv is None: nv = self.nv
        # return init_guess_by_chkfile(self.cell, chk, project)
        return init_guess_by_chkfile(self.cell, chk, nv, project=project)
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

    def get_init_guess(self, key="hcore", nv=None, fC_ks=None):
        if nv is None: nv = self.nv

        if key in ["h1e","hcore","cycle1"]:
            C_ks, mocc_ks = get_init_guess(self, nv=nv, key=key, fC_ks=fC_ks)
        else:
            logger.warn(self, "Unknown init guess %s. Use hcore initial guess",
                        key)
            C_ks, mocc_ks = get_init_guess(self, nv=nv, key="hcore",
                                           fC_ks=fC_ks)

        return C_ks, mocc_ks

    def get_vpplocR(self, mesh=None, Gv=None):
        return pw_helper.get_vpplocR(self.cell, mesh=mesh, Gv=Gv)

    def get_vj_R(self, C_ks, mocc_ks, mesh=None, Gv=None):
        return pw_helper.get_vj_R(self.cell, C_ks, mocc_ks, mesh=mesh, Gv=Gv)

    def scf(self, C0_ks=None, **kwargs):
        self.dump_flags()

        if isinstance(self._acexi_to_save, tempfile._TemporaryFileWrapper):
            facexi = lib.H5TmpFile(self._acexi_to_save.name)
        else:
            facexi = self._acexi_to_save
        self.converged, self.e_tot, self.mo_energy, self.mo_coeff, \
                self.mo_occ = kernel_doubleloop(self, self.kpts,
                           C0_ks=C0_ks, facexi=facexi,
                           nbandv=self.nv, nbandv_extra=self.nv_extra,
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
    dump_moe = dump_moe
    get_init_guess = get_init_guess
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
    mf.kernel()

    mf.dump_scf_summary()
