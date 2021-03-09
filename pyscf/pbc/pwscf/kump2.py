""" kpt-sampled periodic MP2 using a plane wave basis and spin-unrestricted HF
"""

import time
import h5py
import tempfile
import numpy as np

from pyscf.pbc.pwscf import kmp2, pw_helper
from pyscf.pbc.pwscf.kuhf import get_spin_component
from pyscf.pbc import tools
from pyscf import lib
from pyscf.lib import logger


def kconserv(kptija, reduce_latvec, kdota):
    tmp = lib.dot(kptija.reshape(1,-1), reduce_latvec) - kdota
    return np.where(abs(tmp - np.rint(tmp)).sum(axis=1)<1e-6)[0][0]


def fill_oovv(oovv, v_ia, Co_kj_R, Cv_kb_R, fac=None):
    r"""
    Math:
        oovv = \sum_G rho_ia^kika(G)*coulG(ki-ka) * rho_jb^kjkb(kptijab-G)
             = \sum_G V_ia^kika(G) * rho_jb^kjkb(kptijab-G)
             = \sum_r V_ia^kika(r)*phase * rho_jb^kjkb(r)
             = \sum_r v_ia^kika(r) * rho_jb^kjkb(r)
    """
    no_i, no_j = oovv.shape[:2]
    for j in range(no_j):
        rho_jb_R = Co_kj_R[j].conj() * Cv_kb_R
        for i in range(no_i):
            oovv[i,j] = lib.dot(v_ia[i], rho_jb_R.T)
    if not fac is None: oovv *= fac

    return oovv


def kernel_dx_(cell, kpts, chkfile_name, summary, nv=None, nv_lst=None):
    """ Compute both direct (d) and exchange (x) contributions together.
    """

    cput0 = (time.clock(), time.time())

    dtype = np.complex128
    dsize = 16

    fchk, C_ks, moe_ks, mocc_ks = kmp2.read_fchk(chkfile_name)

    nkpts = len(kpts)
    mesh = cell.mesh
    coords = cell.get_uniform_grids(mesh=mesh)
    ngrids = coords.shape[0]

    reduce_latvec = cell.lattice_vectors() / (2*np.pi)
    kdota = lib.dot(kpts, reduce_latvec)

    fac = ngrids**2. / cell.vol
    fac_oovv = fac * ngrids / nkpts

    no_ks = np.asarray([pw_helper.get_no_ks_from_mocc(mocc_ks[s])
                       for s in [0,1]])
    if nv is None:
        n_ks = np.asarray([[len(mocc_ks[s][k]) for k in range(nkpts)]
                          for s in [0,1]])
        nv_ks = n_ks - no_ks
    else:
        if isinstance(nv,int): nv = [nv] * 2
        nv_ks = np.asarray([[nv[s]] * nkpts for s in [0,1]])
        n_ks = no_ks + nv_ks
    no_max = np.max(no_ks)
    nv_max = np.max(nv_ks)
    no_sps = np.asarray([[no_ks[0][k],no_ks[1][k]] for k in range(nkpts)])
    nv_sps = np.asarray([[nv_ks[0][k],nv_ks[1][k]] for k in range(nkpts)])
    n_sps = np.asarray([[n_ks[0][k],n_ks[1][k]] for k in range(nkpts)])
    if nv_lst is None:
        nv_lst = [nv_max]
    nv_lst = np.asarray(nv_lst)
    nnv = len(nv_lst)
    logger.info(cell, "Compute emp2 for these nv's: %s", nv_lst)

    # estimate memory requirement
    est_mem = no_max*nv_max*ngrids      # for caching v_ia_R
    est_mem += (no_max*nv_max)**2*4     # for caching oovv_ka/kb, eijab, wijab
    est_mem += (no_max+nv_max)*ngrids*2 # for caching MOs
    est_mem *= dsize / 1e6
    frac = 0.6
    cur_mem = cell.max_memory - lib.current_memory()[0]
    safe_mem = cur_mem * frac
    logger.debug(cell, "Currently available memory %9.2f MB, safe %9.2f MB",
                 cur_mem, safe_mem)
    logger.debug(cell, "Estimated required memory  %9.2f MB", est_mem)
    if est_mem > safe_mem:
        rec_mem = est_mem / frac + lib.current_memory()[0]
        logger.warn(cell, "Estimate memory requirement (%.2f MB) exceeds %.0f%% of currently available memory (%.2f MB). Calculations may fail and `cell.max_memory = %.2f` is recommended.", est_mem, frac*100, safe_mem, rec_mem)

    buf1 = np.empty(no_max*nv_max*ngrids, dtype=dtype)
    buf2 = np.empty(no_max*no_max*nv_max*nv_max, dtype=dtype)
    buf3 = np.empty(no_max*no_max*nv_max*nv_max, dtype=dtype)

    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    fswap = lib.H5TmpFile(swapfile.name)
    swapfile = None

# ifft to make C(G) --> C(r)
# note the ordering of spin and k-pt indices is swapped
    C_ks_R = fswap.create_group("C_ks_R")
    for s in [0,1]:
        C_ks_s = get_spin_component(C_ks, s)
        for k in range(nkpts):
            key = "%d"%k
            C_ks_R["%s/%d"%(key,s)] = tools.ifft(C_ks_s[key][()], mesh)

    v_ia_ks_R = fswap.create_group("v_ia_ks_R")

    cput1 = logger.timer(cell, 'initialize pwmp2', *cput0)

    tick = np.zeros(2)
    tock = np.zeros(2)
    tspans = np.zeros((7,2))
    tcomps = summary["tcomps"] = ["init", "v_ks_R", "khelper", "IO", "oovv",
                                  "energy", "tot"]
    tspans[0] = np.asarray(cput1) - np.asarray(cput0)

    emp2_d = np.zeros(nnv)
    emp2_x = np.zeros(nnv)
    emp2_ss = np.zeros(nnv)
    emp2_os = np.zeros(nnv)
    for ki in range(nkpts):
        kpti = kpts[ki]
        no_i = no_sps[ki]

        tick[:] = time.clock(), time.time()

        Co_ki_R = [C_ks_R["%d/%d"%(ki,s)][:no_i[s]] for s in [0,1]]

        for ka in range(nkpts):
            kpta = kpts[ka]
            no_a = no_sps[ka]
            nv_a = nv_sps[ka]
            coulG = tools.get_coulG(cell, kpta-kpti, exx=False, mesh=mesh)

            key_ka = "%d"%ka
            if key_ka in v_ia_ks_R: del v_ia_ks_R[key_ka]

            for s in [0,1]:
                Cv_ka_R = C_ks_R["%s/%d"%(key_ka,s)][no_a[s]:no_a[s]+nv_a[s]]
                v_ia_R = np.ndarray((no_i[s],nv_a[s],ngrids), dtype=dtype,
                                    buffer=buf1)

                for i in range(no_i[s]):
                    v_ia = tools.fft(Co_ki_R[s][i].conj() *
                                     Cv_ka_R, mesh) * coulG
                    v_ia_R[i] = tools.ifft(v_ia, mesh)

                v_ia_ks_R["%s/%d"%(key_ka,s)] = v_ia_R
                v_ia_R = Cv_ka_R = None

        Co_ki_R = None

        tock[:] = time.clock(), time.time()
        tspans[1] += tock - tick

        for kj in range(nkpts):
            no_j = no_sps[kj]
            kptij = kpti + kpts[kj]

            tick[:] = time.clock(), time.time()

            Co_kj_R = [C_ks_R["%d/%d"%(kj,s)][:no_j[s]] for s in [0,1]]

            tock[:] = time.clock(), time.time()
            tspans[3] += tock - tick

            done = [False] * nkpts
            kab_lst = []
            kptijab_lst = []
            for ka in range(nkpts):
                if done[ka]: continue
                kptija = kptij - kpts[ka]
                kb = kconserv(kptija, reduce_latvec, kdota)
                kab_lst.append((ka,kb))
                kptijab_lst.append(kptija-kpts[kb])
                done[ka] = done[kb] = True

            tick[:] = time.clock(), time.time()
            tspans[2] += tick - tock

            nkab = len(kab_lst)
            for ikab in range(nkab):
                ka,kb = kab_lst[ikab]
                kptijab = kptijab_lst[ikab]

                no_a = no_sps[ka]
                nv_a = nv_sps[ka]
                no_b = no_sps[kb]
                nv_b = nv_sps[kb]

                tick[:] = time.clock(), time.time()
                phase = np.exp(-1j*lib.dot(coords,
                                           kptijab.reshape(-1,1))).reshape(-1)
                tock[:] = time.clock(), time.time()
                tspans[4] += tock - tick

                for s in [0,1]:

                    tick[:] = time.clock(), time.time()
                    Cv_kb_R = C_ks_R["%d/%d"%(kb,s)][no_b[s]:no_b[s]+nv_b[s]]
                    v_ia = v_ia_ks_R["%d/%d"%(ka,s)][:]
                    tock[:] = time.clock(), time.time()
                    tspans[3] += tock - tick

                    v_ia *= phase
                    oovv_ka = np.ndarray((no_i[s],no_j[s],nv_a[s],nv_b[s]),
                                         dtype=dtype, buffer=buf2)
                    fill_oovv(oovv_ka, v_ia, Co_kj_R[s], Cv_kb_R, fac_oovv)
                    tick[:] = time.clock(), time.time()
                    tspans[4] += tick - tock

                    Cv_kb_R = None

                    if ka != kb:
                        Cv_ka_R = C_ks_R["%d/%d"%(ka,s)][no_a[s]:
                                                         no_a[s]+nv_a[s]]
                        v_ib = v_ia_ks_R["%d/%s"%(kb,s)][:]
                        tock[:] = time.clock(), time.time()
                        tspans[3] += tock - tick

                        v_ib *= phase
                        oovv_kb = np.ndarray((no_i[s],no_j[s],nv_b[s],nv_a[s]),
                                             dtype=dtype, buffer=buf3)
                        fill_oovv(oovv_kb, v_ib, Co_kj_R[s], Cv_ka_R, fac_oovv)
                        tick[:] = time.clock(), time.time()
                        tspans[4] += tick - tock

                        Cv_ka_R = v_ib = None
                    else:
                        oovv_kb = oovv_ka

# Same-spin contribution to KUMP2 energy
                    tick[:] = time.clock(), time.time()
                    mo_e_o = moe_ks[s][ki][:no_i[s]]
                    mo_e_v = moe_ks[s][ka][no_a[s]:no_a[s]+nv_a[s]]
                    eia = mo_e_o[:,None] - mo_e_v

                    if ka != kb:
                        mo_e_o = moe_ks[s][kj][:no_j[s]]
                        mo_e_v = moe_ks[s][kb][no_b[s]:no_b[s]+nv_b[s]]
                        ejb = mo_e_o[:,None] - mo_e_v
                    else:
                        ejb = eia

                    eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                    t2_ijab = np.conj(oovv_ka/eijab)
                    for inv_,nv_ in enumerate(nv_lst):
                        eijab_d = np.einsum('ijab,ijab->',
                                            t2_ijab[:,:,:nv_,:nv_],
                                            oovv_ka[:,:,:nv_,:nv_]).real
                        eijab_x = - np.einsum('ijab,ijba->',
                                              t2_ijab[:,:,:nv_,:nv_],
                                              oovv_kb[:,:,:nv_,:nv_]).real
                        if ka != kb:
                            eijab_d *= 2
                            eijab_x *= 2
                        emp2_d[inv_] += eijab_d
                        emp2_x[inv_] += eijab_x
                        emp2_ss[inv_] += eijab_d + eijab_x
                    tock[:] = time.clock(), time.time()
                    tspans[5] += tock - tick

                    oovv_ka = oovv_kb = eijab = None

# Opposite-spin contribution to KUMP2 energy
                    if s == 0:
                        t = 1 - s
                        tick[:] = time.clock(), time.time()
                        Cv_kb_R = C_ks_R["%d/%d"%(kb,t)][no_b[t]:
                                                         no_b[t]+nv_b[t]]
                        tock[:] = time.clock(), time.time()
                        tspans[3] += tock - tick

                        oovv_ka = np.ndarray((no_i[s],no_j[t],nv_a[s],nv_b[t]),
                                             dtype=dtype, buffer=buf2)
                        fill_oovv(oovv_ka, v_ia, Co_kj_R[t], Cv_kb_R, fac_oovv)
                        tick[:] = time.clock(), time.time()
                        tspans[4] += tick - tock

                        Cv_kb_R = v_ia = None

                        mo_e_o = moe_ks[t][kj][:no_j[t]]
                        mo_e_v = moe_ks[t][kb][no_b[t]:no_b[t]+nv_b[t]]
                        ejb = mo_e_o[:,None] - mo_e_v

                        eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                        t2_ijab = np.conj(oovv_ka/eijab)
                        for inv_,nv_ in enumerate(nv_lst):
                            eijab_d = np.einsum('ijab,ijab->',
                                                t2_ijab[:,:,:nv_,:nv_],
                                                oovv_ka[:,:,:nv_,:nv_]).real
                            if ka != kb:
                                eijab_d *= 2
                            eijab_d *= 2    # alpha,beta <-> beta,alpha
                            emp2_d[inv_] += eijab_d
                            emp2_os[inv_] += eijab_d
                        tock[:] = time.clock(), time.time()
                        tspans[5] += tock - tick

                        oovv_ka = eijab = None
                    else:
                        v_ia = None

        cput1 = logger.timer(cell, 'kpt %d (%6.3f %6.3f %6.3f)'%(ki,*kpti),
                             *cput1)

    buf1 = buf2 = buf3 = None

    emp2_d *= 0.5 / nkpts
    emp2_x *= 0.5 / nkpts
    emp2_ss *= 0.5 / nkpts
    emp2_os *= 0.5 / nkpts
    emp2 = emp2_d + emp2_x
    summary["e_corr_d"] = emp2_d[-1]
    summary["e_corr_x"] = emp2_x[-1]
    summary["e_corr_ss"] = emp2_ss[-1]
    summary["e_corr_os"] = emp2_os[-1]
    summary["e_corr"] = emp2[-1]
    summary["nv_lst"] = nv_lst
    summary["e_corr_d_lst"] = emp2_d
    summary["e_corr_x_lst"] = emp2_x
    summary["e_corr_ss_lst"] = emp2_ss
    summary["e_corr_os_lst"] = emp2_os
    summary["e_corr_lst"] = emp2

    cput1 = logger.timer(cell, 'pwmp2', *cput0)
    tspans[6] = np.asarray(cput1) - np.asarray(cput0)
    for tspan, tcomp in zip(tspans,tcomps):
        summary["t-%s"%tcomp] = tspan

    return emp2[-1]


class PWKUMP2(kmp2.PWKRMP2):
    def __init__(self, mf, nv=None):
        kmp2.PWKRMP2.__init__(self, mf, nv=nv)

    def kernel(self, nv=None, nv_lst=None):
        cell = self.cell
        kpts = self.kpts
        chkfile = self._scf.chkfile
        summary = self.mp2_summary
        if nv is None: nv = self.nv

        self.e_corr = kernel_dx_(cell, kpts, chkfile, summary, nv=nv,
                                 nv_lst=nv_lst)

        self._finalize()

        return self.e_corr


if __name__ == "__main__":
    from pyscf.pbc import gto, scf, mp, pwscf

    atom = "H 0 0 0; H 0.9 0 0"
    a = np.eye(3) * 3
    basis = "gth-szv"
    pseudo = "gth-pade"

    ke_cutoff = 50

    cell = gto.Cell(atom=atom, a=a, basis=basis, pseudo=pseudo,
                    ke_cutoff=ke_cutoff)
    cell.build()
    cell.verbose = 5

    nk = 2
    kmesh = [nk] * 3
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    pwmf = pwscf.PWKUHF(cell, kpts)
    pwmf.nv = 5
    pwmf.kernel()

    es = {"5": -0.01363871}

    pwmp = PWKUMP2(pwmf)
    pwmp.kernel(nv_lst=[5])
    pwmp.dump_mp2_summary()
    nv_lst = pwmp.mp2_summary["nv_lst"]
    ecorr_lst = pwmp.mp2_summary["e_corr_lst"]
    for nv,ecorr in zip(nv_lst,ecorr_lst):
        err = abs(ecorr - es["%d"%nv])
        print(err)
        assert(err < 1e-6)
