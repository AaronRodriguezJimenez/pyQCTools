from pyscf import gto, scf, mcscf
import numpy as np

mol = gto.Mole()
mol.output = 'N2-1.5_triplet.out'
mol.atom = [
        ['N', (0.0, 0.0, -0.75)],
        ['N', (0, 0, 0.75)],
    ]
mol.basis = 'sto-3g'
mol.symmetry = 'D2h' # D2h is more numerically robust than Dooh
mol.build()

mf = scf.RHF(mol)
mf.max_cycle = 100
mf.conv_tol = 1e-8
ehf_val = mf.scf()

mc = mcscf.CASSCF(mf, 8, (6,4)) #Triplet state
mc.fcisolver.conv_tol = 1e-7
mc.fcisolver.threads = 1

ncas = {
    'Ag': 2,  # 2s sigma_g and 2pz sigma_g
    'B1u': 2,  # 2s sigma_u and 2pz sigma_u
    'B2u': 1,  # 2px pi_u
    'B3u': 1,  # 2py pi_u
    'B2g': 1,  # 2px pi_g*
    'B3g': 1  # 2py pi_g*
}

ncore = {
            'Ag': 1,  # 1s sigma_g
            'B1u': 1  # 1s sigma_u
        }

mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas, ncore)
e_tot, _, ci, mo, _ = mc.kernel(mo)

print("CASSCF Energy:")
print(e_tot)