"""
 Here we compute canonical and localized orbitals for Ethylene and compare
 their plots

"""

from pyscf import gto, scf, lo
from pyscf.tools import cubegen
import pyqctools as pq


# 1. Define Molecule
Ethyl = pq.ethylene()
water = pq.H2O()

mol = gto.M(atom=Ethyl, basis='sto-3g')
#mol = gto.M(atom=water, basis='sto-3g')

# 2. Run Restricted Hartree-Fock (RHF)
mf = scf.RHF(mol).run()

print("Canonical orbital coefficients:")
print(mf.mo_coeff[:,1])
print("Total orbitals:", len(mf.mo_coeff))
# C matrix stores the AO to localized orbital coefficients
C = lo.boys.BF(mol, mf.mo_coeff)
C.kernel()

print("Localized orbital coefficients:")
print(C.mo_coeff[:,1])

cubegen.orbital(mol, 'Eth_mo1.cube', mf.mo_coeff[:,0], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo2.cube', mf.mo_coeff[:,1], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo3.cube', mf.mo_coeff[:,2], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo4.cube', mf.mo_coeff[:,3], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo5.cube', mf.mo_coeff[:,4], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo6.cube', mf.mo_coeff[:,5], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo7.cube', mf.mo_coeff[:,6], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo8.cube', mf.mo_coeff[:,7], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo9.cube', mf.mo_coeff[:,8], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo10.cube', mf.mo_coeff[:,9], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo11.cube', mf.mo_coeff[:,10], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo12.cube', mf.mo_coeff[:,11], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo13.cube', mf.mo_coeff[:,12], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo14.cube', mf.mo_coeff[:,13], nx=80, ny=80, nz=80)

cubegen.orbital(mol, 'Eth_mo1_loc.cube', C.mo_coeff[:,0], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo2_loc.cube', C.mo_coeff[:,1], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo3_loc.cube', C.mo_coeff[:,2], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo4_loc.cube', C.mo_coeff[:,3], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo5_loc.cube', C.mo_coeff[:,4], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo6_loc.cube', C.mo_coeff[:,5], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo7_loc.cube', C.mo_coeff[:,6], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo8_loc.cube', C.mo_coeff[:,7], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo9_loc.cube', C.mo_coeff[:,8], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo10_loc.cube', C.mo_coeff[:,9], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo11_loc.cube', C.mo_coeff[:,10], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo12_loc.cube', C.mo_coeff[:,11], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo13_loc.cube', C.mo_coeff[:,12], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'Eth_mo14_loc.cube', C.mo_coeff[:,13], nx=80, ny=80, nz=80)

exit()
#= = =  CANONICAL = = =
cubegen.orbital(mol, 'h2o_mo1.cube', mf.mo_coeff[:,0], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'h2o_mo2.cube', mf.mo_coeff[:,1], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'h2o_mo3.cube', mf.mo_coeff[:,2], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'h2o_mo4.cube', mf.mo_coeff[:,3], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'h2o_mo5.cube', mf.mo_coeff[:,4], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'h2o_mo6.cube', mf.mo_coeff[:,5], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'h2o_mo7.cube', mf.mo_coeff[:,6], nx=80, ny=80, nz=80)

#= = = LOCALIZED
cubegen.orbital(mol, 'h2o_mo1_loc.cube', C.mo_coeff[:,0], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'h2o_mo2_loc.cube', C.mo_coeff[:,1], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'h2o_mo3_loc.cube', C.mo_coeff[:,2], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'h2o_mo4_loc.cube', C.mo_coeff[:,3], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'h2o_mo5_loc.cube', C.mo_coeff[:,4], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'h2o_mo6_loc.cube', C.mo_coeff[:,5], nx=80, ny=80, nz=80)
cubegen.orbital(mol, 'h2o_mo7_loc.cube', C.mo_coeff[:,6], nx=80, ny=80, nz=80)