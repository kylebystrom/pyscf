#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pyscf import scf, gto
from pyscf.eph import eph_fd, uhf
import numpy as np
import unittest

mol = gto.M()
mol.atom = [['O', [0.000000000000,  -0.000000000775,   0.923671924285]],
            ['H', [-0.000000000000,  -1.432564848017,   2.125164039823]],
            ['H', [0.000000000000,   1.432564848792,   2.125164035930]]]

mol.unit = 'Bohr'
mol.basis = 'sto3g'
mol.verbose=4
mol.build() # this is a pre-computed relaxed geometry

class KnownValues(unittest.TestCase):
    def test_finite_diff_uhf_eph(self):
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-16
        mf.conv_tol_grad = 1e-10
        mf.kernel()

        grad = mf.nuc_grad_method().kernel()
        self.assertTrue(abs(grad).max()<1e-5)
        mat, omega = eph_fd.kernel(mf)
        matmo, _ = eph_fd.kernel(mf, mo_rep=True)

        myeph = uhf.EPH(mf)
        eph, _ = myeph.kernel()
        ephmo, _ = myeph.kernel(mo_rep=True)
        for i in range(len(omega)):
            self.assertTrue(min(np.linalg.norm(eph[:,i]-mat[:,i]),np.linalg.norm(eph[:,i]+mat[:,i]))<1e-5)
            self.assertTrue(min(abs(eph[:,i]-mat[:,i]).max(), abs(eph[:,i]+mat[:,i]).max())<1e-5)
            self.assertTrue(min(np.linalg.norm(ephmo[:,i]-matmo[:,i]),np.linalg.norm(ephmo[:,i]+matmo[:,i]))<1e-5)
            self.assertTrue(min(abs(ephmo[:,i]-matmo[:,i]).max(), abs(ephmo[:,i]+matmo[:,i]).max())<1e-5)

if __name__ == '__main__':
    print("Full Tests for UHF")
    unittest.main()
