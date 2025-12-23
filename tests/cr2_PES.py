'''
Scan Cr2 molecule singlet state dissociation curve.

Based on the need to control the CASSCF initial
guess using functions project_init_guess and sort_mo.

In this example:
sort_mo function is replaced by the symmetry-adapted version
``sort_mo_by_irrep``.

Total electrons:  48
Active electrons: 12
Core electrons:   36
Core orbitals:    18

Using Dooh symetry
# Total 18 core orbitals
ncore = {'A1g': 4, 'A1u': 4, 'E1ux': 2, 'E1uy': 2, 'E1gx': 2, 'E1gy': 2}
ncas = {'A1g':2, 'A1u':2,
        'E1ux':1, 'E1uy':1, 'E1gx':1, 'E1gy':1,
        'E2ux':1, 'E2uy':1, 'E2gx':1, 'E2gy':1}


Using D2h symmetry (here)
# ... inside run function ...
# D2h mapping for (12e, 12o)
# 3 Ag (s-sigma, d-sigma, d-delta)
# 3 B1u (s-sigma*, d-sigma*, d-delta*)
# 1 B2g/B3g (d-pi-g)
# 1 B2u/B3u (d-pi-u)
# 1 B1g (d-delta-g)
# 1 Au (d-delta-u)
'''

import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import symm

ehf = []
emc = []

def run_proj(b, dm, mo, ci=None):
    mol = gto.Mole()
    mol.verbose = 4
    mol.output = 'cr2-%2.1f.out' % b
    mol.atom = [
        ['Cr', (0.0, 0.0, -b/2)],
        ['Cr', (0, 0, b/2)],
    ]
    mol.basis = 'sto-3g'
    mol.symmetry = 'D2h' # D2h is more numerically robust than Dooh
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 100
    mf.conv_tol = 1e-8

    # Use the previous density matrix to speed up SCF convergence
    ehf_val = mf.scf(dm)
    ehf.append(ehf_val)

    mc = mcscf.CASSCF(mf, 12, 12)
    mc.fcisolver.conv_tol = 1e-7
    mc.fcisolver.threads = 1

    if mo is None:
        # --- INITIAL POINT SETUP ---
        # Map 12 valence orbitals (3d, 4s) to D2h irreps
        ncas = {'Ag': 3, 'B1u': 3, 'B2g': 1, 'B3g': 1, 'B2u': 1, 'B3u': 1, 'B1g': 1, 'Au': 1}
        # 18 core orbitals (36 electrons)
        ncore = {'Ag': 4, 'B1u': 4, 'B2g': 2, 'B3g': 2, 'B2u': 3, 'B3u': 3}

        mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas, ncore)
    else:
        # --- PROJECTION STEP ---
        # Project MOs from the previous geometry onto the current basis
        mo = mcscf.project_init_guess(mc, mo)

        # Run CASSCF using projected MOs and previous CI vector
        # Kernel returns: Total Energy, E_cas, CI_vector, MO_coeffs, MO_energies
    e_tot, _, ci, mo, _ = mc.kernel(mo, ci)
    emc.append(e_tot)

    mc.analyze()

    # Return DM for next RHF, and MO/CI for next CASSCF
    return mf.make_rdm1(), mo, ci

def run(b, dm, mo, ci=None):
    symm.geom.TOLERANCE = 1e-2

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'cr2-%2.1f.out' % b
    mol.atom = [
        ['Cr',(  0.000000,  0.000000, -b/2)],
        ['Cr',(  0.000000,  0.000000,  b/2)],
    ]
    mol.basis = 'sto-3g' #'cc-pVTZ'
    mol.symmetry = 'D2h'# 1
    mol.build()
    mf = scf.RHF(mol)
    mf.level_shift = .4
    mf.max_cycle = 100
    mf.conv_tol = 1e-6
    ehf.append(mf.scf(dm))

    mc = mcscf.CASSCF(mf, 12, 12)
    mc.fcisolver.conv_tol = 1e-6
    # FCI solver with multi-threads is not stable enough for this sytem
    mc.fcisolver.threads = 1
    if mo is None:
        # the initial guess for b = 1.5, symetry-aware initial guess in Dooh
        #ncore = {'A1g':5, 'A1u':5}
        #ncore = {'A1g': 4, 'A1u': 4, 'E1ux': 2, 'E1uy': 2, 'E1gx': 2, 'E1gy': 2}
        #ncas = {'A1g':2, 'A1u':2,
        #        'E1ux':1, 'E1uy':1, 'E1gx':1, 'E1gy':1,
        #        'E2ux':1, 'E2uy':1, 'E2gx':1, 'E2gy':1}

        # initial guess for b=1.5, in D2h
        ncore = {'Ag': 4, 'B1u': 4, 'B2g': 2, 'B3g': 2, 'B2u': 3, 'B3u': 3}
        ncas = {'Ag': 3, 'B1u': 3, 'B2g': 1, 'B3g': 1, 'B2u': 1, 'B3u': 1, 'B1g': 1, 'Au': 1}

        mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas, ncore)
    else:
        # Don't pass symmetry labeled mo across different bond lenghts;
        mo = mcscf.project_init_guess(mc, mo)

    emc.append(mc.kernel(mo, ci)[0])
    mc.analyze()
    return mf.make_rdm1(), mc.mo_coeff, mc.ci

# --- Scan Logic ---
dm = mo = ci = None

# Forward scan
for b in numpy.arange(1.5, 3.01, .1):
    dm, mo, ci = run_proj(b, dm, mo, ci)

# Backward scan (re-uses last dm/mo/ci from the end of the forward scan)
for b in reversed(numpy.arange(1.5, 3.01, .1)):
    dm, mo, ci = run_proj(b, dm, mo, ci)

# --- Data Processing and Plotting ---
x = numpy.arange(1.5, 3.01, .1)
ehf1 = ehf[:len(x)]
ehf2 = ehf[len(x):]
emc1 = emc[:len(x)]
emc2 = emc[len(x):]
ehf2.reverse()
emc2.reverse()
with open('cr2-scan.txt', 'w') as fout:
    fout.write('     HF 1.5->3.0     CAS(12,12)      HF 3.0->1.5     CAS(12,12)\n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f  %12.8f\n'
                   % (xi, ehf1[i], emc1[i], ehf2[i], emc2[i]))

import matplotlib.pyplot as plt
plt.plot(x, ehf1, label='HF,1.5->3.0')
plt.plot(x, ehf2, label='HF,3.0->1.5')
plt.plot(x, emc1, label='CAS(12,12),1.5->3.0')
plt.plot(x, emc2, label='CAS(12,12),3.0->1.5')
plt.legend()
plt.show()
