""" Check PW occ + CPW vir for MP2
"""


import h5py
import tempfile
import numpy as np

from pyscf.pbc import gto, pwscf
from pyscf import lib


if __name__ == "__main__":
    kmesh = [2,1,1]
    ke_cutoff = 30
    basis_cpw = "ccecp-cc-pvdz"
    pseudo = "gth-pade"
    atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994"
    a = np.asarray(
        [[0.       , 1.78339987, 1.78339987],
        [1.78339987, 0.        , 1.78339987],
        [1.78339987, 1.78339987, 0.        ]])

# cell
    cell = gto.Cell(
        atom=atom,
        a=a,
        basis="gth-szv",
        pseudo=pseudo,
        ke_cutoff=ke_cutoff
    )
    cell.build()
    cell.verbose = 6

# kpts
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

# HF
    mf = pwscf.KRHF(cell, kpts)
    mf.kernel()

    assert(abs(mf.e_tot - -10.6754924867542) < 1.e-6)

# MP2
    moe_ks, mocc_ks = mf.get_cpw_virtual(basis_cpw)
    mf.dump_moe(moe_ks, mocc_ks)
    mmp = pwscf.KMP2(mf)
    mmp.kernel()
    mmp.dump_mp2_summary()

    assert(abs(mmp.e_corr - -0.215895180360867) < 1.e-6)
