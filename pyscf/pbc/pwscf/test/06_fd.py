import unittest
import tempfile
import numpy as np
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.pwscf import khf, kuhf, krks, kuks
import pyscf.pbc
from numpy.testing import assert_allclose
pyscf.pbc.DEBUG = False


def setUpModule():
    global CELL, KPTS, ATOM, KPT1
    CELL = pbcgto.Cell(
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994",
        a = np.asarray([
                [0.       , 1.78339987, 1.78339987],
                [1.78339987, 0.        , 1.78339987],
                [1.78339987, 1.78339987, 0.        ]]),
        basis="gth-szv",
        ke_cutoff=50,
        pseudo="gth-pade",
    )
    CELL.mesh = [13, 13, 13]
    # CELL.mesh = [27, 27, 27]
    CELL.build()

    kmesh = [3, 1, 1]
    KPTS = CELL.make_kpts(kmesh)

    ATOM = pbcgto.Cell(
        atom = "C 0 0 0",
        a = np.eye(3) * 4,
        basis="gth-szv",
        ke_cutoff=50,
        pseudo="gth-pade",
        spin=2,
    )
    ATOM.mesh = [25, 25, 25]
    ATOM.build()
    ATOM.verbose = 6

    nk = 1
    kmesh = (nk,)*3
    KPT1 = ATOM.make_kpts(kmesh)


def tearDownModule():
    global CELL, ATOM
    del CELL, ATOM


class KnownValues(unittest.TestCase):

    def _get_calc(self, cell, kpts, spinpol=False, xc=None, run=True, **kwargs):
        if xc is None:
            if not spinpol:
                mf = khf.PWKRHF(cell, kpts)
            else:
                mf = kuhf.PWKUHF(cell, kpts)
        else:
            if not spinpol:
                mf = krks.PWKRKS(cell, kpts, xc=xc)
            else:
                mf = kuks.PWKUKS(cell, kpts, xc=xc)
        mf.__dict__.update(**kwargs)
        if run:
            mf.kernel()
        return mf

    def _check_rhf_uhf(self, cell, kpts, xc=None, rtol=1e-7, atol=1e-7):
        rmf = self._get_calc(cell, kpts, spinpol=False, xc=xc)
        umf = self._get_calc(cell, kpts, spinpol=True, xc=xc)
        assert_almost_equal(rmf.e_tot, umf.e_tot, rtol=rtol, atol=atol)

    def _check_fd(self, mf):
        if not mf.converged:
            mf.kernel()
            assert mv.converged
        mo_energy, mo_occ = mf.get_mo_energy(mf.mo_coeff, mf.mo_occ)
        nkpts = len(mf.mo_coeff)
        # C_ks = [C.copy() for C in mf.mo_coeff]
        delta = 1e-5
        cell = mf.cell
        mesh = cell.mesh
        Gv = cell.get_Gv(mesh)
    
        spinpol = isinstance(mf, kuhf.PWKUHF)
        def _update(Ct_ks):
            mf.update_pp(Ct_ks)
            mf.update_k(Ct_ks, mo_occ)

        def _transform(C_ks, mocc_ks, k):
            vbm = np.max(np.where(mocc_ks[k] > 0.9))
            cbm = np.min(np.where(mocc_ks[k] < 0.1))
            transform = np.identity(C_ks[k].shape[0])
            transform[vbm, vbm] = np.sqrt(0.5)
            transform[vbm, cbm] = -np.sqrt(0.5)
            transform[cbm, cbm] = np.sqrt(0.5)
            transform[cbm, vbm] = np.sqrt(0.5)
            Ct_k = transform.dot(C_ks[k])
            Ct_ks = [C_k.copy() for C_k in C_ks]
            Ct_ks[k] = Ct_k.copy()
            return Ct_ks, vbm, cbm

        for k in range(nkpts):
            Ct_ks, vbm, cbm = _transform(mf.mo_coeff, mo_occ, k)
            _update(Ct_ks)

            Ctt_ks, moett_ks, mocctt_ks = mf.eig_subspace([C.copy() for C in Ct_ks], mo_occ, Gv=Gv, mesh=mesh)
            ham1 = np.einsum("ig,jg->ij", Ctt_ks[k], Ct_ks[k].conj())
            ham2 = np.einsum("ki,i,ij->kj", ham1.conj().T, moett_ks[k], ham1)

            new_ham = mf.get_mo_energy(Ct_ks, mo_occ, full_ham=True)
            expected_de = new_ham[k][vbm, cbm] + new_ham[k][cbm, vbm]
            if hasattr(mf, "xc") and not mf._numint.libxc.is_hybrid_xc(mf.xc):
                assert_allclose(ham2[vbm, cbm] + ham2[cbm, vbm], expected_de)

            vj_R = mf.get_vj_R(Ct_ks, mo_occ)
            new_vbm = Ct_ks[k][vbm].copy()
            new_cbm = Ct_ks[k][cbm].copy()
            new_vbm_p = new_vbm + 0.5 * delta * new_cbm
            new_vbm_m = new_vbm - 0.5 * delta * new_cbm

            Ct_ks[k][vbm] = new_vbm_m
            mf.update_pp(Ct_ks)
            mf.update_k(Ct_ks, mo_occ)
            vj_R = mf.get_vj_R(Ct_ks, mo_occ, mesh=mesh, Gv=Gv)
            em = mf.energy_elec([C.copy() for C in Ct_ks], mo_occ, Gv=Gv, mesh=mesh, vj_R=vj_R)

            Ct_ks[k][vbm] = new_vbm_p
            mf.update_pp(Ct_ks)
            mf.update_k(Ct_ks, mo_occ)
            vj_R = mf.get_vj_R(Ct_ks, mo_occ, mesh=mesh, Gv=Gv)
            ep = mf.energy_elec([C.copy() for C in Ct_ks], mo_occ)
            fd = (ep - em) / delta

            # NOTE need to understand the factor of 2 a bit better
            # but the factor of nkpts is just because the fd energy
            # is per unit cell, but the gap is the energy derivative
            # for the supercell with respect to perturbing the orbital
            expected_de = expected_de * 2 / nkpts
            print(expected_de, fd)
            assert_allclose(expected_de, fd, atol=1e-8, rtol=1e-8)

    def test_fd_rhf(self):
        mf = self._get_calc(CELL, KPTS, nvir=2)
        self._check_fd(mf)

    def _check_fd_rks(self, xc, mesh=None):
        if mesh is None:
            cell = CELL
        else:
            cell = CELL.copy()
            cell.mesh = mesh
            cell.build()
        mf = self._get_calc(cell, KPTS, nvir=2, xc=xc, damp_type="simple", damp_factor=0.7)
        self._check_fd(mf)

    def test_fd_rks_lda(self):
        self._check_fd_rks("LDA")

    def test_fd_rks_gga(self):
        self._check_fd_rks("PBE")

    def test_fd_rks_mgga(self):
        self._check_fd_rks("R2SCAN", mesh=[17, 17, 17])


if __name__ == "__main__":
    print("Finite difference for pbc.pwscf -- khf, kuhf, krks, kuks")
    unittest.main()

