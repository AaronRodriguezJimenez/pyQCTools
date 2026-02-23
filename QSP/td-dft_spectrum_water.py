from pyscf import gto, dft, tddft
import numpy as np
import matplotlib.pyplot as plt
from pyqctools.geometries import H2O

# 1. Define molecule
mol = gto.Mole()
mol.atom = H2O()
mol.basis = "sto3g"
mol.symmetry = True
mol.verbose = 4
mol.output = "./H2O.out"
mol.build()

# 2. Run Ground State Density Functional Theory (DFT)
mf = dft.RKS(mol)
mf.xc = 'b3lyp' # Use a functional suitable for excited states
mf.kernel()

# 3. Run TDDFT for Excited States
# nstates=10 asks for the first 10 excited states
td_df = tddft.TDDFT(mf)
td_df.nstates = 10
td_df.kernel()

# 4. Analyze Results
# Excitation energies in eV
print("Excitation Energies (eV):", td_df.e * 27.2114)
# Oscillator strengths (intensity)
print("Oscillator Strengths:", td_df.oscillator_strength())

# 5. Simple Plotting (Broadening with Gaussian)
energies = td_df.e * 27.2114
intensities = td_df.oscillator_strength()
broadening = 0.2 # eV
x = np.linspace(0, 25, 1000)
y = np.zeros_like(x)

for e, i in zip(energies, intensities):
    y += i * np.exp(-(x - e)**2 / (2 * broadening**2))

plt.plot(x, y)
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.title('UV-Vis Spectrum of H2O')
plt.show()
