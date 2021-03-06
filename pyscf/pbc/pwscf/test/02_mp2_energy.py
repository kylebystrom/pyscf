""" Check if the PW code gives same MO energies as the GTO code for a given wave function
"""


import h5py
import tempfile
import numpy as np

from pyscf.pbc import gto, scf, pwscf, mp
from pyscf.pbc.pwscf import khf, pw_helper
from pyscf import lib
import pyscf.lib.parameters as param


if __name__ == "__main__":
    nk = 2
    kmesh = (nk,) * 3
    ke_cutoff = 100
    pseudo = "gth-pade"
    exxdiv = "ewald"
    atom = "H 0 0 0; H 0.9 0 0"
    a = np.eye(3) * 3

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

    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

# GTO
    gmf = scf.KRHF(cell, kpts)
    gmf.exxdiv = exxdiv
    gmf.kernel()

    gmp = mp.KMP2(gmf)
    gmp.kernel()

# PW
    pmf = pwscf.KRHF(cell, kpts)
    pmf.exxdiv = exxdiv
    n_ks = [gmf.mo_coeff[0].shape[1]] * nkpts
    C_ks = pw_helper.get_C_ks_G(cell, kpts, gmf.mo_coeff, n_ks)
    mocc_ks = khf.get_mo_occ(cell, C_ks=C_ks)
    pmf.nv = np.sum(mocc_ks[0]<1e-3)
    mesh = cell.mesh
    Gv = cell.get_Gv(mesh)
    vpplocR = pmf.get_vpplocR()
    vj_R = pmf.get_vj_R(C_ks, mocc_ks)

    # init ACE
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    facexi = lib.H5TmpFile(swapfile.name)
    swapfile = None
    ace_xi_ks = facexi.create_group("ace_xi_ks")
    pmf.initialize_ACE(C_ks, mocc_ks, kpts, mesh, Gv, ace_xi_ks=ace_xi_ks,
                       Ct_ks=C_ks)

    moe_ks = pmf.get_mo_energy(C_ks, mocc_ks, mesh, Gv, C_ks_exx=C_ks,
                               ace_xi_ks=ace_xi_ks,
                               vpplocR=vpplocR, vj_R=vj_R)[0]

    # fake a scf chkfile
    swapfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
    pmf.chkfile = swapfile.name
    swapfile = None
    pwscf.chkfile.dump_scf(cell, pmf.chkfile, 0., moe_ks, mocc_ks, C_ks)

    from pyscf.pbc.pwscf import kmp2
    pmp = kmp2.PWKRMP2(pmf)
    pmp.kernel()

    print(gmp.e_corr)
    print(pmp.e_corr)

    assert(abs(gmp.e_corr - pmp.e_corr) < 1.e-6)
