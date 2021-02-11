""" Check if the PW code gives same MO energies as the GTO code for a given wave function
"""


import numpy as np

from pyscf.pbc import gto, df, scf, pwscf
from pyscf.pbc.pwscf import pw_helper
from pyscf import lib
import pyscf.lib.parameters as param


if __name__ == "__main__":
    nk = 2
    kmesh = (nk,) * 3
    ke_cutoff = 80
    pseudo = "gth-pade"
    exxdiv = None
    atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994"
    a = np.asarray(
        [[0.       , 1.78339987, 1.78339987],
        [1.78339987, 0.        , 1.78339987],
        [1.78339987, 1.78339987, 0.        ]])

    # first run a GTO calculation
    gcell = gto.Cell(
        atom=atom,
        a=a,
        basis="gth-szv",
        pseudo=pseudo,
        ke_cutoff=ke_cutoff
    )
    gcell.build()
    gcell.verbose = 5

    kpts = gcell.make_kpts(kmesh)

    gmf = scf.KRHF(gcell, kpts)
    gmf.exxdiv = exxdiv
    gmf.kernel()

    # get each components of the mo energy
    vpp = lib.asarray(gmf.with_df.get_pp(kpts))
    vkin = lib.asarray(gmf.cell.pbc_intor('int1e_kin', 1, 1, kpts))
    dm = gmf.make_rdm1()
    vj, vk = df.FFTDF(gcell).get_jk(dm, kpts=kpts)

    nkpts = len(kpts)
    moe_comp_ks = np.zeros((4,nkpts), dtype=np.complex128)
    for ik in range(nkpts):
        moe_comp_ks[0,ik] = np.einsum("ij,ji->", vkin[ik], dm[ik])
        moe_comp_ks[1,ik] = np.einsum("ij,ji->", vpp[ik], dm[ik])
        moe_comp_ks[2,ik] = np.einsum("ij,ji->", vj[ik], dm[ik]) * 0.5
        moe_comp_ks[3,ik] = -np.einsum("ij,ji->", vk[ik], dm[ik]) * 0.25

    # PW
    pcell = gto.Cell(
        atom=atom,
        a=a,
        basis="PW",
        ke_cutoff=ke_cutoff,
        pseudo=pseudo
    )
    pcell.build()

    pmf = pwscf.KRHF(pcell, kpts)
    pmf.exxdiv = exxdiv
    C_ks = pw_helper.get_Co_ks_G(gcell, kpts, gmf.mo_coeff, gmf.mo_occ)
    vpplocR = pmf.get_vpplocR()
    vj_R = pmf.get_vj_R(C_ks)
    moe_comp_ks_pw = np.zeros((4, nkpts), dtype=np.complex128)
    for ik in range(nkpts):
        moe = pmf.apply_Fock_kpt(C_ks[ik], kpts[ik], C_ks,
                                 vpplocR=vpplocR, vj_R=vj_R, ret_E=True)[1]
        moe_comp_ks_pw[0,ik] = moe[0]
        moe_comp_ks_pw[1,ik] = moe[1] + moe[2]
        moe_comp_ks_pw[2:,ik] = moe[3:]

    maxe_real = np.max(np.abs(moe_comp_ks.real - moe_comp_ks_pw.real))
    maxe_imag = np.max(np.abs(moe_comp_ks.imag - moe_comp_ks_pw.imag))
    print(maxe_real, maxe_imag)

    assert(maxe_real < 1e-3)
    assert(maxe_imag < 1e-3)
