import h5py
import numpy as np

from pyscf.pbc import cc
from pyscf.pbc.pwscf.ao2mo.molint import get_molint_from_C
from pyscf.pbc.pwscf.khf import THR_OCC


class PWKRCCSD:
    def __init__(self, mf):
        self.mf = mf

    def kernel(self):
        eris = _ERIS(self.mf)
        mcc = cc.kccsd_rhf.RCCSD(self.mf)
        mcc.kernel(eris=eris)

        return mcc


class _ERIS:
    def __init__(self, mf):
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ

        cell = mf.cell
        kpts = mf.kpts
        nkpts = len(kpts)

        self.e_hf = mf.e_tot
        self.mo_energy = np.asarray(mo_energy)
# remove ewald correction
        moe_noewald = np.zeros_like(self.mo_energy)
        for k in range(nkpts):
            moe = self.mo_energy[k].copy()
            moe[mo_occ[k]>THR_OCC] += mf._madelung
            moe_noewald[k] = moe
        self.fock = np.asarray([np.diag(moe.astype(np.complex128)) for moe in moe_noewald])

        with h5py.File(mf.chkfile, "r") as f:
            mo_coeff = f["mo_coeff"]
            eris = get_molint_from_C(cell, mo_coeff,
                                     kpts).transpose(0,2,1,3,5,4,6)

        no = np.sum(mo_occ[0]>THR_OCC)
        self.oooo = eris[:,:,:,:no,:no,:no,:no]
        self.ooov = eris[:,:,:,:no,:no,:no,no:]
        self.oovv = eris[:,:,:,:no,:no,no:,no:]
        self.ovov = eris[:,:,:,:no,no:,:no,no:]
        self.voov = eris[:,:,:,no:,:no,:no,no:]
        self.vovv = eris[:,:,:,no:,:no,no:,no:]
        self.vvvv = eris[:,:,:,no:,no:,no:,no:]

        eris = None


if __name__ == "__main__":
    a0 = 1.78339987
    atom = "C 0 0 0; C %.10f %.10f %.10f" % (a0*0.5, a0*0.5, a0*0.5)
    a = np.asarray(
        [[0., a0, a0],
        [a0, 0., a0],
        [a0, a0, 0.]])

    from pyscf.pbc import gto, scf, pwscf
    cell = gto.Cell(atom=atom, a=a, basis="gth-szv", pseudo="gth-pade",
                    ke_cutoff=50)
    cell.build()
    cell.verbose = 5

    kpts = cell.make_kpts([2,1,1])

    mf = scf.KRHF(cell, kpts)
    mf.kernel()

    mcc = cc.kccsd_rhf.RCCSD(mf)
    mcc.kernel()

    from pyscf.pbc.pwscf.pw_helper import gtomf2pwmf
    pwmf = gtomf2pwmf(mf)
    pwmcc = PWKRCCSD(pwmf).kernel()

    assert(np.abs(mcc.e_corr - pwmcc.e_corr) < 1e-5)
