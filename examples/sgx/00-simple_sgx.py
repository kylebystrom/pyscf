#!/usr/bin/env python

'''
This example shows how to use pseudo spectral integrals in SCF calculation.
'''

from pyscf import gto
from pyscf import scf
from pyscf import sgx
import numpy as np
mol = gto.M(
    atom='''O    0.   0.       0.
            H    0.   -0.757   0.587
            H    0.   0.757    0.587
    ''',
    basis = 'def2-qzvppd',
    verbose=4
)
mf = scf.RHF(mol)
mf.kernel()
import time
start = time.monotonic()
mf = sgx.sgx_fit(scf.RHF(mol), pjs=False)
mf.with_df.grids_level_i = 1
mf.with_df.grids_level_f = 1
mf.with_df.dfj = True
mf.conv_tol = 1e-9
mf.kernel()
end = time.monotonic()
print(end-start)
#print(mf.with_df._opt.dm_cond)
#print(mf.with_df._opt._this.contents)
#print(np.sum(mf.with_df._opt.dm_cond < 1e-7))

# Using RI for Coulomb matrix while K-matrix is constructed with COS-X method
#mf.with_df.dfj = True
#mf.kernel()

#mf = sgx.sgx_fit(scf.RHF(mol), pjs=True)
#mf.kernel() 
