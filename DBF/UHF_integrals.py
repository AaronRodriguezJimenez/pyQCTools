"""
 Molecular integrals from UHF
"""
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import lo
from pyscf import ao2mo
import os

#- - - Tensor functions

def save_tensors(b, h0, h1a, h1b, eri_aa, eri_bb, eri_ab, folder="tensors"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Filename includes the bond distance for easy identification
    filename = os.path.join(folder, f"{b:2.1f}_tensors.npz")

    # Save arrays; h0 is stored as a 0-d array
    print("H0 = ", h0)
    np.savez_compressed(filename, hc=h0, h1_a=h1a, h1_b=h1b,
                        h2aa=eri_aa, h2bb=eri_bb, h2ab=eri_ab)
    print(f"Tensors saved to {filename}")

def get_integrals(mol, mf, localized=False, debug=True):
    """
    Robustly extract active-space tensors supporting UHF (and RHF fallback).
    Accepts multiple mo_coeff / mo_occ layouts (tuple/list, 2D, or 3D with spin axis).

    Returns:
      h0, h1_alpha, h1_beta, h2_aa, h2_bb, h2_ab
    Optionally also Ca_final, Cb_final when return_mos=True.

    Set debug=True to print shapes during execution for easier debugging.
    """
    # AO & nuclear
    h0 = mol.energy_nuc()

    # AO integrals
    h_ao = mf.get_hcore()  # one-electron AO h_core (shape: nAO x nAO)
    s_ao = mol.intor('int1e_ovlp')  # overlap (nAO x nAO)
    eri_ao = mol.intor('int2e')  # two-electron AO integrals (pq|rs) shape: (nAO,nAO,nAO,nAO)

    n_ao = h_ao.shape[0]

    # get mo_coeff and mo_occ robustly
    mo_coeff = getattr(mf, 'mo_coeff', None)
    if mo_coeff is None:
        raise ValueError("mf.mo_coeff is None")

    Ca, Cb = mf.mo_coeff

    # number of MOs
    nmo_a = Ca.shape[1]
    nmo_b = Cb.shape[1]

    # count occupied orbitals
    a_nocc = int((mf.mo_occ[0] > 0).sum())
    b_nocc = int((mf.mo_occ[1] > 0).sum())

    if debug:
        print("DEBUG shapes:")
        print("  hcore:", h_ao.shape)
        print("  eri.shape:", getattr(eri_ao, 'shape', None))
        print("  Ca.shape:", Ca.shape)
        print("  Cb.shape:", Cb.shape)
        print("  a_nocc:", a_nocc)
        print("  b_nocc:", b_nocc)
        #print("  occ_a.shape:", occ_a.shape)
        #print("  occ_b.shape:", occ_b.shape)

    # Localization (Boys) separately for occ/vir, per spin
    if localized:
        a_occ_idx = np.arange(0, a_nocc)
        a_vir_idx = np.arange(a_nocc, nmo_a)
        b_occ_idx = np.arange(0, b_nocc)
        b_vir_idx = np.arange(b_nocc, nmo_b)

        Ca_occ = Ca[:, a_occ_idx] if a_occ_idx.size > 0 else Ca[:, :0]
        Ca_vir = Ca[:, a_vir_idx] if a_vir_idx.size > 0 else Ca[:, :0]
        Cb_occ = Cb[:, b_occ_idx] if b_occ_idx.size > 0 else Cb[:, :0]
        Cb_vir = Cb[:, b_vir_idx] if b_vir_idx.size > 0 else Cb[:, :0]

        a_loc_occ = lo.Boys(mol, Ca_occ).kernel() if Ca_occ.shape[1] >= 2 else Ca_occ.copy()
        a_loc_vir = lo.Boys(mol, Ca_vir).kernel() if Ca_vir.shape[1] >= 2 else Ca_vir.copy()
        b_loc_occ = lo.Boys(mol, Cb_occ).kernel() if Cb_occ.shape[1] >= 2 else Cb_occ.copy()
        b_loc_vir = lo.Boys(mol, Cb_vir).kernel() if Cb_vir.shape[1] >= 2 else Cb_vir.copy()

        Ca_final = np.column_stack((a_loc_occ, a_loc_vir)) if (a_loc_occ.size + a_loc_vir.size) > 0 else Ca.copy()
        Cb_final = np.column_stack((b_loc_occ, b_loc_vir)) if (b_loc_occ.size + b_loc_vir.size) > 0 else Cb.copy()
    else:
        Ca_final = Ca.copy()
        Cb_final = Cb.copy()

    # sanity checks
    if Ca_final.shape[0] != n_ao or Cb_final.shape[0] != n_ao:
        raise ValueError(f"Final MO coeff matrices must have {n_ao} rows (AO basis). Got {Ca_final.shape}, {Cb_final.shape}")

    # one-body in MO basis
    h1_alpha = Ca_final.T @ h_ao @ Ca_final
    h1_beta  = Cb_final.T @ h_ao @ Cb_final

    # two-electron integrals
    # eri_ao has indices (p,q,r,s)
#    eri_mo_aaaa = np.einsum('pqrs,pi,qj,rk,sl->ijkl',eri_ao, Ca, Ca, Ca, Ca, optimize=True)
#    ri_bbbb = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao, Cb, Cb, Cb, Cb, optimize=True)
#    eri_aabb = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao, Ca, Ca, Cb, Cb, optimize=True)

    h2_aa = ao2mo.kernel(mol, (Ca, Ca, Ca, Ca))
    h2_aa = ao2mo.restore(1, h2_aa, n_ao)

    h2_bb = ao2mo.kernel(mol,(Cb, Cb, Cb, Cb))
    h2_bb = ao2mo.restore(1, h2_bb, n_ao)

    h2_ab = ao2mo.kernel(mol, (Ca_final, Ca_final, Cb_final, Cb_final))
    h2_ab = ao2mo.restore(1, h2_ab, n_ao)

    return h0, h1_alpha, h1_beta, h2_aa, h2_bb, h2_ab

b = 1.1
mol = gto.Mole()
mol.atom = [
        ['N', (0.0, 0.0, -b/2)],
        ['N', (0, 0, b/2)]]

mol.basis = 'sto-3g'
   # mol.symmetry = 'D2h' # D2h is more numerically robust than Dooh
mol.spin = 0
mol.build()
mf = scf.UHF(mol)
mf.max_cycle = 100
mf.conv_tol = 1e-8
mf.run()
ss = mf.spin_square()
print("E(UHF) = ", mf.e_tot)
print(f"Spin square <S^2>: {ss[0]}")

h0, h1a, h1b, eri_aa, eri_bb, eri_ab = get_integrals(mol, mf, localized=False)
dir = "N2_test_uhf"

save_tensors(b, h0, h1a, h1b, eri_aa, eri_bb, eri_ab, dir)
"""
HF energy formula:
E_elec =  sum_{i in occ_a} h1_alpha[i,i] + sum_{i in occ_b} h1_beta[i,i]
        + 1/2 * ( sum_{i,j in occ_a} [ h2_aa[i,i,j,j] - h2_aa[i,j,j,i] ]
                + sum_{i,j in occ_b} [ h2_bb[i,i,j,j] - h2_bb[i,j,j,i] ] )
        +       ( sum_{i in occ_a, j in occ_b} h2_ab[i,i,j,j] )
"""