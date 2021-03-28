""" Check PW-KRHF, PW-KRMP2 and read init guess from chkfile
"""


import h5py
import tempfile
import numpy as np

from pyscf.pbc import gto, pwscf
from pyscf import lib


if __name__ == "__main__":
    kmesh = [2,1,1]
    ke_cutoff = 50
    pseudo = None
    atom = "He 0 0 0"
    a = np.eye(3) * 2

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
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

# tempfile
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    chkfile = swapfile.name
    swapfile = None

# krhf
    pwmf = pwscf.KRHF(cell, kpts)
    pwmf.nvir = 10 # request 10 virtual states
    pwmf.chkfile = chkfile
    pwmf.kernel()

    assert(abs(pwmf.e_tot - -3.03580640722391) < 1.e-4)

# krhf init from chkfile
    pwmf.init_guess = "chkfile"
    pwmf.kernel()

# input C0
    with h5py.File(pwmf.chkfile, "r") as f:
        C0 = [f["mo_coeff/%d"%k][()] for k in range(nkpts)]
    pwmf.kernel(C0=C0)

# krmp2
    pwmp = pwscf.KMP2(pwmf)
    pwmp.kernel()

    assert(abs(pwmp.e_corr - -0.0183110607989341) < 1.e-4)
