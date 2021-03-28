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
    pmf = pwscf.KUHF(cell, kpts)
    pmf.init_pp()
    pmf.init_jk()
    pmf.exxdiv = exxdiv
    C_ks = [None] * 2
    for s in [0,1]:
        n_ks = [gmf.mo_coeff[0].shape[1]] * nkpts
        C_ks[s] = pw_helper.get_C_ks_G(cell, kpts, gmf.mo_coeff, n_ks)
    mocc_ks = kuhf.get_mo_occ(cell, C_ks=C_ks)
    pmf.update_pp(C_ks)
    pmf.update_k(C_ks, mocc_ks)
    vj_R = pmf.get_vj_R(C_ks, mocc_ks)
    pmf.nvir = [np.sum(mocc_ks[s][0]<1e-3) for s in [0,1]]

    moe_ks = pmf.get_mo_energy(C_ks, mocc_ks, exxdiv=exxdiv, vj_R=vj_R,
                               ret_mocc=False)

    # fake a scf chkfile
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    pmf.chkfile = swapfile.name
    swapfile = None
    pwscf.chkfile.dump_scf(cell, pmf.chkfile, 0., moe_ks, mocc_ks, C_ks)

    from pyscf.pbc.pwscf import kump2
    pmp = kump2.PWKUMP2(pmf)
    pmp.kernel()

    print(pmp.e_corr)
    print(gmp.e_corr)

    assert(abs(gmp.e_corr - pmp.e_corr) < 1.e-6)
