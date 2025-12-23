#
# PES of N2 comparing RHF, CCSD(t) and FCI
#
from pyscf import gto, scf, cc, fci
from pyscf.scf import rhf_symm
from matplotlib import pyplot as plt

import pyqctools as pq
import numpy as np

# distances
distances = np.arange(0.5, 3.6, 0.1)
fci_energies = []
ccsd_t_energies = []
rhf_energies = []

print(f"Distance,  HF: , CCSD(t): , FCI(t): ")
for d in distances:
    #coord = pq.N2(d)
    coord = pq.H2(d)

    mol = gto.M(atom=coord,  # Ethyl,
                basis='3-21g',
                spin=0,
                cart=True,
                verbose=0)

    # 2. Run Restricted Hartree-Fock (RHF)
    # - - - Normal SCF - - -
    mf = scf.RHF(mol)
    e_hf = mf.kernel()

    # 3. Initialize and run CCSD(t) calculation
    mycc = cc.CCSD(mf).run()
    e_ccsd = mycc.e_tot
    et = mycc.ccsd_t()

    # - - - FCI - - -
    myhf = mol.RHF().run()
    # create an FCI solver based on the SCF object
    cisolver = fci.FCI(myhf)
    e_fci = cisolver.kernel()[0]

    # Print results
    print(f"{d}, {e_hf}, {mycc.e_tot + et}, {e_fci}")
    #append results
    fci_energies.append(e_fci)
    ccsd_t_energies.append(mycc.e_tot + et)
    rhf_energies.append(e_hf)

rhf_energies = np.array(rhf_energies)
ccsd_t_energies = np.array(ccsd_t_energies)
fci_energies = np.array(fci_energies)

# Plot results
plt.figure()
plt.plot(distances, rhf_energies, 'k.-', label='RHF')
plt.plot(distances, ccsd_t_energies, 'b-', label='CCSD(T)')
plt.plot(distances, fci_energies, 'r.-', label='FCI')
plt.title("N2/STO-3G")
plt.xlabel("Distance")
plt.ylabel("Energy (a.u.")

plt.legend()
plt.show()