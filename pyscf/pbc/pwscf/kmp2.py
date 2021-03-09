""" kpt-sampled periodic MP2 using a plane wave basis
"""

import time
import h5py
import tempfile
import numpy as np

from pyscf.pbc.pwscf import pw_helper
from pyscf.pbc import tools
from pyscf import lib
from pyscf.lib import logger


def read_fchk(chkfile_name):
    from pyscf.lib.chkfile import load
    scf_dict = load(chkfile_name, "scf")
    mocc_ks = scf_dict["mo_occ"]
    moe_ks = scf_dict["mo_energy"]
    scf_dict = None

    fchk = h5py.File(chkfile_name, "a")
    C_ks = fchk["mo_coeff"]

    return fchk, C_ks, moe_ks, mocc_ks


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

    Args:
        nv_lst (array-like of int):
            If given, the MP2 correlation energies using the number of virtual orbitals specified by the list will be returned.
    """

    cput0 = (time.clock(), time.time())

    dtype = np.complex128
    dsize = 16

    fchk, C_ks, moe_ks, mocc_ks = read_fchk(chkfile_name)

    nkpts = len(kpts)
    mesh = cell.mesh
    coords = cell.get_uniform_grids(mesh=mesh)
    ngrids = coords.shape[0]

    reduce_latvec = cell.lattice_vectors() / (2*np.pi)
    kdota = lib.dot(kpts, reduce_latvec)

    fac = ngrids**2. / cell.vol
    fac_oovv = fac * ngrids / nkpts

    no_ks = pw_helper.get_no_ks_from_mocc(mocc_ks)
    if nv is None:
        n_ks = [len(mocc_ks[k]) for k in range(nkpts)]
        nv_ks = [n_ks[k] - no_ks[k] for k in range(nkpts)]
    else:
        nv_ks = [nv] * nkpts
        n_ks = [no_ks[k] + nv_ks[k] for k in range(nkpts)]
    no_max = np.max(no_ks)
    nv_max = np.max(nv_ks)
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

    C_ks_R = fswap.create_group("C_ks_R")
    for k in range(nkpts):
        key = "%d"%k
        C_ks_R[key] = tools.ifft(C_ks[key][()], mesh)

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
        no_i = no_ks[ki]

        tick[:] = time.clock(), time.time()

        Co_ki_R = C_ks_R["%d"%ki][:no_i]

        for ka in range(nkpts):
            kpta = kpts[ka]
            no_a = no_ks[ka]
            nv_a = nv_ks[ka]
            coulG = tools.get_coulG(cell, kpta-kpti, exx=False, mesh=mesh)

            Cv_ka_R = C_ks_R["%d"%ka][no_a:no_a+nv_a]
            v_ia_R = np.ndarray((no_i,nv_a,ngrids), dtype=dtype, buffer=buf1)

            for i in range(no_i):
                v_ia = tools.fft(Co_ki_R[i].conj() * Cv_ka_R, mesh) * coulG
                v_ia_R[i] = tools.ifft(v_ia, mesh)

            key = "%d"%ka
            if key in v_ia_ks_R: del v_ia_ks_R[key]
            v_ia_ks_R[key] = v_ia_R
            v_ia_R = Cv_ka_R = None

        Co_ki_R = None

        tock[:] = time.clock(), time.time()
        tspans[1] += tock - tick

        for kj in range(nkpts):
            no_j = no_ks[kj]
            kptij = kpti + kpts[kj]

            tick[:] = time.clock(), time.time()

            Co_kj_R = C_ks_R["%d"%kj][:no_j]

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

                no_a = no_ks[ka]
                nv_a = nv_ks[ka]
                no_b = no_ks[kb]
                nv_b = nv_ks[kb]

                tick[:] = time.clock(), time.time()
                Cv_kb_R = C_ks_R["%d"%kb][no_b:no_b+nv_b]
                v_ia = v_ia_ks_R["%d"%ka][:]
                tock[:] = time.clock(), time.time()
                tspans[3] += tock - tick

                phase = np.exp(-1j*lib.dot(coords,
                                           kptijab.reshape(-1,1))).reshape(-1)
                v_ia *= phase
                oovv_ka = np.ndarray((no_i,no_j,nv_a,nv_b), dtype=dtype,
                                     buffer=buf2)
                fill_oovv(oovv_ka, v_ia, Co_kj_R, Cv_kb_R, fac_oovv)
                tick[:] = time.clock(), time.time()
                tspans[4] += tick - tock

                Cv_kb_R = v_ia = None

                if ka != kb:
                    Cv_ka_R = C_ks_R["%d"%ka][no_a:no_a+nv_a]
                    v_ib = v_ia_ks_R["%d"%kb][:]
                    tock[:] = time.clock(), time.time()
                    tspans[3] += tock - tick

                    v_ib *= phase
                    oovv_kb = np.ndarray((no_i,no_j,nv_b,nv_a), dtype=dtype,
                                         buffer=buf3)
                    fill_oovv(oovv_kb, v_ib, Co_kj_R, Cv_ka_R, fac_oovv)
                    tick[:] = time.clock(), time.time()
                    tspans[4] += tick - tock

                    Cv_ka_R = v_ib = None
                else:
                    oovv_kb = oovv_ka

# KMP2 energy evaluation starts here
                tick[:] = time.clock(), time.time()
                mo_e_o = moe_ks[ki][:no_i]
                mo_e_v = moe_ks[ka][no_a:no_a+nv_a]
                eia = mo_e_o[:,None] - mo_e_v

                if ka != kb:
                    mo_e_o = moe_ks[kj][:no_j]
                    mo_e_v = moe_ks[kb][no_b:no_b+nv_b]
                    ejb = mo_e_o[:,None] - mo_e_v
                else:
                    ejb = eia

                eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
                t2_ijab = np.conj(oovv_ka/eijab)

                for inv_,nv_ in enumerate(nv_lst):
                    eijab_d = 2 * np.einsum('ijab,ijab->',
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
                    emp2_ss[inv_] += eijab_d * 0.5 + eijab_x
                    emp2_os[inv_] += eijab_d * 0.5

                tock[:] = time.clock(), time.time()
                tspans[5] += tock - tick

                oovv_ka = oovv_kb = eijab = woovv = None

        cput1 = logger.timer(cell, 'kpt %d (%6.3f %6.3f %6.3f)'%(ki,*kpti),
                             *cput1)

    buf1 = buf2 = buf3 = None

    emp2_d /= nkpts
    emp2_x /= nkpts
    emp2_ss /= nkpts
    emp2_os /= nkpts
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


class PWKRMP2:
    def __init__(self, mf, nv=None):
        self.cell = self.mol = mf.cell
        self._scf = mf

        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.nv = nv

##################################################
# don't modify the following attributes, they are not input options
        self.kpts = mf.kpts
        self.nkpts = len(self.kpts)
        self.mp2_summary = dict()
        self.e_hf = self._scf.e_tot
        self.e_corr = None
        self.t2 = None
        self._keys = set(self.__dict__.keys())

    @property
    def e_tot(self):
        if self.e_corr is None:
            return None
        else:
            return self.e_hf + self.e_corr

    def dump_mp2_summary(self, verbose=logger.DEBUG):
        log = logger.new_logger(self, verbose)
        summary = self.mp2_summary
        def write(fmt, key):
            if key in summary:
                log.info(fmt, summary[key])
        log.info('**** MP2 Summaries ****')
        log.info('Number of virtuals =              %d', summary["nv_lst"][-1])
        log.info('Total Energy (HF+MP2) =           %24.15f', self.e_tot)
        log.info('Correlation Energy =              %24.15f', self.e_corr)
        write('Direct Energy =                   %24.15f', 'e_corr_d')
        write('Exchange Energy =                 %24.15f', 'e_corr_x')
        write('Same-spin Energy =                %24.15f', 'e_corr_ss')
        write('Opposite-spin Energy =            %24.15f', 'e_corr_os')

        nv_lst = summary["nv_lst"]
        if len(nv_lst) > 1:
            log.info('%sNvirt  Ecorr', "\n")
            ecorr_lst = summary["e_corr_lst"]
            for nv,ecorr in zip(nv_lst,ecorr_lst):
                log.info("%5d  %24.15f", nv, ecorr)
            log.info("%s", "")

        def write_time(comp, t_comp, t_tot):
            tc, tw = t_comp
            tct, twt = t_tot
            rc = tc / tct * 100
            rw = tw / twt * 100
            log.info('CPU time for %10s %9.2f  ( %6.2f%% ), wall time %9.2f  ( %6.2f%% )', comp.ljust(10), tc, rc, tw, rw)

        t_tot = summary["t-tot"]
        for icomp,comp in enumerate(summary["tcomps"]):
            write_time(comp, summary["t-%s"%comp], t_tot)

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

    def _finalize(self):
        logger.note(self, "KMP2 energy = %.15g", self.e_corr)


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

    pwmf = pwscf.PWKRHF(cell, kpts)
    pwmf.nv = 20
    pwmf.kernel()

    es = {"5": -0.01363871, "10": -0.01873622, "20": -0.02461560}

    pwmp = PWKRMP2(pwmf)
    pwmp.kernel(nv_lst=[5,10,20])
    pwmp.dump_mp2_summary()
    nv_lst = pwmp.mp2_summary["nv_lst"]
    ecorr_lst = pwmp.mp2_summary["e_corr_lst"]
    for nv,ecorr in zip(nv_lst,ecorr_lst):
        err = abs(ecorr - es["%d"%nv])
        print(err)
        assert(err < 1e-6)
