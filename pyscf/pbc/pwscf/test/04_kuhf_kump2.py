""" Check PW-KUHF, PW-KUMP2 and read init guess from chkfile
"""


import tempfile
import numpy as np

from pyscf.pbc import gto, pwscf
from pyscf import lib


if __name__ == "__main__":
    nk = 1
    ke_cutoff = 50
    pseudo = "gth-pade"
    atom = "C 0 0 0"
    a = np.eye(3) * 6   # atom in a cubic box

# cell
    cell = gto.Cell(
        atom=atom,
        a=a,
        spin=2, # triplet
        basis="gth-szv",
        pseudo=pseudo,
        ke_cutoff=ke_cutoff
    )
    cell.build()
    cell.verbose = 5

# kpts
    kmesh = [nk]*3
    kpts = cell.make_kpts(kmesh)

# tempfile
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    chkfile = swapfile.name
    swapfile = None

# krhf
    pwmf = pwscf.KUHF(cell, kpts)
    pwmf.nv = 4 # request 4 virtual states
    pwmf.chkfile = chkfile
    pwmf.kernel()

    assert(abs(pwmf.e_tot - -5.36567883224574) < 1.e-4)

# krhf init from chkfile
    pwmf.init_guess = "chkfile"
    pwmf.kernel()

# krmp2
    # pwmp = pwscf.KUP2(pwmf)
    # pwmp.kernel()
    #
    # assert(abs(pwmp.e_corr - -0.172159082251055) < 1.e-4)
