from pyscf import gto, dft, tddft
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the molecule
mol = gto.Mole()
mol.atom = '''
  C     0.0000000    1.3970000    0.0000000
  C     1.2099600    0.6985000    0.0000000
  C     1.2099600   -0.6985000    0.0000000
  C     0.0000000   -1.3970000    0.0000000
  C    -1.2099600   -0.6985000    0.0000000
  C    -1.2099600    0.6985000    0.0000000
  H     0.0000000    2.4810000    0.0000000
  H     2.1495000    1.2410000    0.0000000
  H     2.1495000   -1.2410000    0.0000000
  H     0.0000000   -2.4810000    0.0000000
  H    -2.1495000   -1.2410000    0.0000000
  H    -2.1495000    1.2410000    0.0000000
'''
mol.basis = 'def2-svp'
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
x = np.linspace(0, 10, 1000)
y = np.zeros_like(x)

for e, i in zip(energies, intensities):
    y += i * np.exp(-(x - e)**2 / (2 * broadening**2))

plt.plot(x, y)
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.title('UV-Vis Spectrum of Benzene')
plt.show()
