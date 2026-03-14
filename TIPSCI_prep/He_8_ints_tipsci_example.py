from pyscf import gto, scf, ao2mo
from orbitalpartitioning import *
from pyscf.tools import molden
import numpy as np
import os

molecule = """
He 0.00 0.00 0.00
He 1.5 0.00 0.00
He 1.5 1.5 0.00
He 0.00 1.5 0.00
He 0.70 0.70 2.20
He 2.20 0.70 2.20
He 2.20 2.20 2.20
He 0.70 2.20 2.20
"""
basis = "6-31g"
pymol = gto.Mole(
        atom    =   molecule,
        symmetry=   True,
        spin    =   0, # number of unpaired electrons
        charge  =   0,
        basis   =   basis)


pymol.build()
print("symmetry: ",pymol.topgroup)
# mf = pyscf.scf.UHF(pymol).x2c()
mf = scf.ROHF(pymol)
mf.verbose = 4
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-5
mf.chkfile = "scf.fchk"
mf.init_guess = "sad"

mf.run(max_cycle=200)

print(" Hartree-Fock Energy: %12.8f" % mf.e_tot)
# mf.analyze()

# Get data
F = mf.get_fock()
C = mf.mo_coeff
S = mf.get_ovlp()
ndocc = mf.nelec[1]
nsing = mf.nelec[0] - ndocc
nvirt = mf.mol.nao - ndocc - nsing

# Just use alpha orbitals
Cdocc = mf.mo_coeff[:,0:ndocc]
Csing = mf.mo_coeff[:,ndocc:ndocc+nsing]
Cvirt = mf.mo_coeff[:,ndocc+nsing:ndocc+nsing+nvirt]

nbas = Cdocc.shape[0]

# Find AO's corresponding to atoms
full = []
frag1 = []
frag2 = []
frag3 = []
for ao_idx, ao in enumerate(mf.mol.ao_labels(fmt=False)):
    if ao[0] == 0:
        frag1.append(ao_idx)
        full.append(ao_idx)
    elif ao[0] == 1:
        frag1.append(ao_idx)
        full.append(ao_idx)
    elif ao[0] == 2:
        frag1.append(ao_idx)
        full.append(ao_idx)
    elif ao[0] == 3:
        frag1.append(ao_idx)
        full.append(ao_idx)
    elif ao[0] == 4:
        frag2.append(ao_idx)
        full.append(ao_idx)
    elif ao[0] == 5:
        frag2.append(ao_idx)
        full.append(ao_idx)
    elif ao[0] == 6:
        frag2.append(ao_idx)
        full.append(ao_idx)
    elif ao[0] == 7:
        frag2.append(ao_idx)
        full.append(ao_idx)

frags = [frag1, frag2]
print(frags)

# Define projectors
X = np.eye(nbas)
X = scipy.linalg.sqrtm(S)
Pfull = X[:,full]  # non-orthogonal
Pf = []
for f in frags:
    Pf.append(X[:,f])

(Oact, Sact, Vact), (Cenv, Cerr, _) = svd_subspace_partitioning((Cdocc, Csing, Cvirt), Pfull, S)
assert(Cerr.shape[1] == 0)
Cact = np.hstack((Oact,Vact))

# Project active orbitals onto fragments
init_fspace = []
clusters = []
Cfrags = []
orb_index = 1



for fi,f in enumerate(frags):
    print()
    print(" Fragment: ", f)
    (Of, Sf, Vf), (_, _, _) = svd_subspace_partitioning((Oact, Sact, Vact), Pf[fi], S)
    Cfrags.append(np.hstack((Of, Sf, Vf)))
    ndocc_f = Of.shape[1]
    init_fspace.append((ndocc_f+Sf.shape[1], ndocc_f))
    nmof = Of.shape[1] + Sf.shape[1] + Vf.shape[1]
    clusters.append(list(range(orb_index, orb_index+nmof)))
    orb_index += nmof



# Orthogonalize Fragment orbitals
Cfrags = sym_ortho(Cfrags, S)

Cact = np.hstack(Cfrags)

# Write Molden files for visualization
molden.from_mo(mf.mol, "Pfull.molden", Pfull)
molden.from_mo(mf.mol, "Cact.molden", Cact)
molden.from_mo(mf.mol, "Cenv.molden", Cenv)

for i in range(len(frags)):
    molden.from_mo(mf.mol, "Cfrag%i.molden"%i, Cfrags[i])

print(" init_fspace: ", init_fspace)
print(" clusters   : ", clusters)

print(Cenv.shape)
print(Cact.shape)
d1_embed = 2 * Cenv @ Cenv.T

h0 = gto.mole.energy_nuc(mf.mol)
h  = scf.hf.get_hcore(mf.mol)
j, k = scf.hf.get_jk(mf.mol, d1_embed, hermi=1)

print(h.shape)

h0 += np.trace(d1_embed @ ( h + .5*j - .25*k))

h = Cact.T @ h @ Cact;
j = Cact.T @ j @ Cact;
k = Cact.T @ k @ Cact;

nact = h.shape[0]

h2 = ao2mo.kernel(pymol, Cact, aosym="s4", compact=False)
h2.shape = (nact, nact, nact, nact)

# The use of d1_embed only really makes sense if it has zero electrons in the
# active space. Let's warn the user if that's not true

S = pymol.intor("int1e_ovlp_sph")
n_act = np.trace(S @ d1_embed @ S @ Cact @ Cact.T)
try:
    abs(n_act) > 1e-8 == False
    print(n_act)
except Exception as error:
    print(" I found embedded electrons in the active space?!")

h1 = h + j - .5*k;

#dir_out = "/Users/admin/PycharmProjects/pyQCTools/ExchCoups"
#from src.pyqctools.int_fcns import save_restricted_integrals
#save_restricted_integrals("He8-RHF", h0, h1, h2, dir_out)

np.savez_compressed("ints_h0.npz", h0=h0)
np.savez_compressed("ints_h1.npz", h1=h1)
np.savez_compressed("ints_h2.npz", h2=h2)
np.savez_compressed("mo_coeffs.npz", Cact=Cact)
np.savez_compressed("overlap_mat.npz", S=S)
