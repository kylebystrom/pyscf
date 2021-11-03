""" Check if the PW code gives same MO energies as the GTO code for a given wave function
"""


import h5py
import tempfile
import numpy as np

from pyscf.pbc import gto, scf, pwscf, mp
from pyscf.pbc.pwscf import kuhf, pw_helper
from pyscf import lib
import pyscf.lib.parameters as param


if __name__ == "__main__":
    kmesh = [1,1,1]
    ke_cutoff = 50
    pseudo = "gth-pade"
    exxdiv = "ewald"
    atom = "C 0 0 0"
    a = np.eye(3) * 4

# cell
    cell = gto.Cell(
        atom=atom,
        a=a,
        basis="gth-szv",
        pseudo=pseudo,
        ke_cutoff=ke_cutoff,
        spin=2
    )
    cell.build()
    cell.verbose = 5

    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

# GTO
    gmf = scf.UHF(cell, kpts)
    gmf.exxdiv = exxdiv
    gmf.kernel()

    # gmp = mp.KMP2(gmf)
    from pyscf import mp
    gmp = mp.UMP2(gmf)
    gmp.kernel()

# PW
    pmf = pw_helper.gtomf2pwmf(gmf)

    from pyscf.pbc.pwscf import kump2
    pmp = kump2.PWKUMP2(pmf)
    pmp.kernel()

    print(pmp.e_corr)
    print(gmp.e_corr)

    assert(abs(gmp.e_corr - pmp.e_corr) < 1.e-6)
