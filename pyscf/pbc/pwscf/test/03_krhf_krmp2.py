""" Check PW-KRHF, PW-KRMP2 and read init guess from chkfile
"""


import tempfile
import numpy as np

from pyscf.pbc import gto, pwscf
from pyscf import lib


if __name__ == "__main__":
    nk = 2
    ke_cutoff = 50
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
    cell.verbose = 5

# kpts
    kmesh = [nk]*3
    kpts = cell.make_kpts(kmesh)

# tempfile
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    chkfile = swapfile.name
    swapfile = None

# krhf
    pwmf = pwscf.KRHF(cell, kpts)
    pwmf.nv = 10 # request 10 virtual states
    pwmf.chkfile = chkfile
    pwmf.kernel()

    assert(abs(pwmf.e_tot - -11.0397030993567) < 1.e-4)

# krhf init from chkfile
    pwmf.init_guess = "chkfile"
    pwmf.kernel()

# krmp2
    pwmp = pwscf.KMP2(pwmf)
    pwmp.kernel()

    assert(abs(pwmp.e_corr - -0.172159082251055) < 1.e-4)
