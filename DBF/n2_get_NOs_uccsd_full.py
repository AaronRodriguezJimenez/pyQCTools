"""
  In this example, we illustrate how to get Active NOS
  CREATES: SPINORBITALS in chemist notation from RHF
  """

import numpy as np
from pyscf import gto
from pyscf import scf, fci, cc
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
    np.savez_compressed(filename, hc=h0, h1_a=h1a, h1_b=h1b,
                        h2aa=eri_aa, h2bb=eri_bb, h2ab=eri_ab)
    print(f"Tensors saved to {filename}")

def get_active_space_tensors(mol, nat_ccsd_coeff, mf, localized=False, ccsd=False, natural=False, debug=True):
    """
    Robustly extract active-space tensors supporting UHF (and RHF fallback).
    Accepts multiple mo_coeff / mo_occ layouts (tuple/list, 2D, or 3D with spin axis).
    Adapted function using natural orbitals ccsd natural orbitals, this must be
    provided with setting ccsd=True, SCF orbitals will be used if uccsd=False
    Returns:
      h0, h1_alpha, h1_beta, h2_aa, h2_bb, h2_ab
    Optionally also Ca_final, Cb_final when return_mos=True.

    Set debug=True to print shapes during execution for easier debugging.
    """
    # AO & nuclear
    ao_kin = mol.intor('int1e_kin', hermi=1)
    ao_nuc = mol.intor('int1e_nuc', hermi=1)
    ao_obi = ao_kin + ao_nuc
    ao_eri = mol.intor('int2e')

    n_ao = ao_obi.shape[0]

    # get mo_coeff and mo_occ robustly
    if ccsd==False:
        mo_coeff = getattr(mf, 'mo_coeff', None)
        if mo_coeff is None:
            raise ValueError("mf.mo_coeff is None")

        Ca, Cb = mo_coeff
    else:
        Ca, Cb = nat_ccsd_coeff


    # number of MOs
    nmo_a = Ca.shape[1]
    nmo_b = Cb.shape[1]

    # count occupied orbitals
    a_nocc = int((mf.mo_occ[0] > 0).sum())
    b_nocc = int((mf.mo_occ[1] > 0).sum())

    if debug:
        print("DEBUG shapes:")
        print("  ao_obi.shape:", ao_obi.shape)
        print("  ao_eri.shape:", getattr(ao_eri, 'shape', None))
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

    elif natural:
        # Create Natural orbitals:
        dm = mf.make_rdm1()
        dma, dmb = dm[0], dm[1]
        S = mf.get_ovlp()

        # Solve generalized eigenproblem D C = S C n
        # eigh returns eigenvalues in ascending order by default
        from scipy.linalg import eigh  # generalized Hermitian eigensolver
        eigvals_a, eigvecs_a = eigh(dma, S)
        eigvals_b, eigvecs_b = eigh(dmb, S)

        # sort descending (largest occupations first)
        idx_a = eigvals_a.argsort()[::-1]
        idx_b = eigvals_b.argsort()[::-1]

        NOs_a = eigvecs_a[:, idx_a]  # columns are AO coefficients of natural orbitals
        NOs_b = eigvecs_b[:, idx_b]  # columns are AO coefficients of natural orbitals

        Ca_final = NOs_a
        Cb_final = NOs_b

        #CHECKS: Orthonormality in AO metric
        #occ_a = eigvals_a[idx_a]
        #occ_b = eigvals_b[idx_b]
        #assert np.allclose(Ca_final.conj().T @ S @ Ca_final, np.eye(Ca_final.shape[1]), atol=1e-10)
        #assert np.allclose(Cb_final.conj().T @ S @ Cb_final, np.eye(Cb_final.shape[1]), atol=1e-10)
        # Density reconstruction in AO basis
        #Da_rec = S @ Ca_final @ np.diag(occ_a) @ Ca_final.conj().T @ S
        #Db_rec = S @ Cb_final @ np.diag(occ_b) @ Cb_final.conj().T @ S
        #assert np.allclose(Da_rec, dma, atol=1e-10)
        #assert np.allclose(Db_rec, dmb, atol=1e-10)

    else:
        Ca_final = Ca.copy()
        Cb_final = Cb.copy()

    # sanity checks
    if Ca_final.shape[0] != n_ao or Cb_final.shape[0] != n_ao:
        raise ValueError(f"Final MO coeff matrices must have {n_ao} rows (AO basis). Got {Ca_final.shape}, {Cb_final.shape}")

    # Nuclear Repulsion Term
    h0 = mol.energy_nuc()
    # one-body in MO basis
    h1_alpha = Ca_final.T @ ao_obi @ Ca_final
    h1_beta  = Cb_final.T @ ao_obi @ Cb_final

    # two-electron integrals
    h2_aa = ao2mo.kernel(mol, (Ca_final, Ca_final, Ca_final, Ca_final))
    h2_aa = ao2mo.restore(1, h2_aa, n_ao)

    h2_bb = ao2mo.kernel(mol, (Cb_final, Cb_final, Cb_final, Cb_final))
    h2_bb = ao2mo.restore(1, h2_bb, n_ao)

    h2_ab = ao2mo.kernel(mol, (Ca_final, Ca_final, Cb_final, Cb_final))
    h2_ab = ao2mo.restore(1, h2_ab, n_ao)

    return h0, h1_alpha, h1_beta, h2_aa, h2_bb, h2_ab


ehf = []
efci = []
def run_proj(b, get_ints):
    mol = gto.Mole()
    mol.verbose = 4
    mol.output = '/Users/admin/PycharmProjects/pyQCTools/DBF/N2_tensors_ccsd/N2-%2.1f.out' % b
    mol.atom = [
        ['N', (0.0, 0.0, -b/2)],
        ['N', (0, 0, b/2)],
    ]
    mol.basis = 'sto-3g'
   # mol.symmetry = 'D2h' # D2h is more numerically robust than Dooh
    mol.spin = 0
    mol.build()

    mf = scf.UHF(mol)
    mf.max_cycle = 100
    mf.conv_tol = 1e-8

    #- - Check SCF Stability
    # Loop for optimizing orbitals until stable
    #
    from pyscf.lib import logger

    def stable_opt_internal(mf, stability_cycles):
        log = logger.new_logger(mf)
        mo1, _, stable, _ = mf.stability(return_status=True)
        cyc = 0
        print("STABLE?", stable)
        while (not stable and cyc < stability_cycles):
            log.note('Try to optimize orbitals until stable, attempt %d' % cyc)
            dm1 = mf.make_rdm1(mo1, mf.mo_occ)
            mf = mf.run(dm1)
            mo1, _, stable, _ = mf.stability(return_status=True)
            cyc += 1
        if not stable:
            log.note('Stability Opt failed after %d attempts' % cyc)
        return mf

    stability_cycles = 10
    mf.run()
    mf = stable_opt_internal(mf, stability_cycles)
    ehf_val = mf.kernel()
    ehf.append(ehf_val)
    print(f"Total UHF energy: {ehf_val}")
    print(f"Number of alpha electrons: {mf.nelec[0]}")
    print(f"Number of beta electrons: {mf.nelec[1]}")

    # 4. Analyze results (e.g., spin contamination)
    # <S^2> should be 3.75 for a quartet (S=3/2, S(S+1)=3.75)
    ss = mf.spin_square()
    print(f"UHF spin square <S^2>: {ss[0]}")

    # 5. Perform UCCSD for improved natural orbitals
    mycc = cc.UCCSD(mf)
    mycc.kernel()
    mycc.solve_lambda()
    rdm1 = mycc.make_rdm1()

    # Containers for results
    no_coeffs = []
    no_occs = []

    # 2. Process each spin channel
    #    Since rdm1 is (28, 28), we use the full mf.mo_coeff (28, 28)
    for spin_idx, (dm, mo_c_full) in enumerate(zip(rdm1, mf.mo_coeff)):
        spin_label = "Alpha" if spin_idx == 0 else "Beta"

        # Diagonalize the FULL 1-RDM
        # This naturally recovers:
        #   - Frozen Core orbitals (Occupancy = 1.0)
        #   - Active Natural Orbitals (Occupancy between 0 and 1)
        occ_vals, vecs = np.linalg.eigh(dm)

        # Sort High -> Low
        idx = occ_vals.argsort()[::-1]
        occ_vals = occ_vals[idx]
        vecs = vecs[:, idx]

        # Transform to AO basis
        # (28, 28) = (28, 28) . (28, 28)
        no_c = np.dot(mo_c_full, vecs)

        no_coeffs.append(no_c)
        no_occs.append(occ_vals)

        print(f"{spin_label} Natural Occupancies (Top 10):")
        print(f"  {occ_vals}")

    # Final objects
    no_coeff_uhf = np.array(no_coeffs)
    #no_occ_uhf = np.array(no_occs)

    # Get spatial integrals
    if get_ints:
        h0, h1a, h1b, eri_aa, eri_bb, eri_ab = get_active_space_tensors(mol, no_coeff_uhf, mf,
                                                                        localized=False,
                                                                        ccsd=True,
                                                                        natural=False)
        dir = "/Users/admin/PycharmProjects/pyQCTools/DBF/N2_tensors_ccsd"
        save_tensors(b, h0, h1a, h1b, eri_aa, eri_bb, eri_ab, dir)


    # - - - FCI - - -
    cisolver = fci.FCI(mf)
    e_fci = cisolver.kernel()[0]
    efci.append(e_fci)
    return


# --- Scan Logic ---
dm = mo = ci = None

# Forward scan
for b in np.arange(0.5, 3.01, .1):
    run_proj(b, get_ints=False)

# Backward scan re-uses last dm/mo/ci from the end of the forward scan
# Collect tensors for JWmapping

for b in reversed(np.arange(0.5, 3.01, .1)):
    run_proj(b, get_ints=True)

# --- Data Processing and Plotting ---
x = np.arange(0.5, 3.01, .1)
ehf1 = ehf[:len(x)]
ehf2 = ehf[len(x):]
emc1 = efci[:len(x)]
emc2 = efci[len(x):]
ehf2.reverse()
emc2.reverse()

with open('/Users/admin/PycharmProjects/pyQCTools/DBF/N2_tensors_ccsd/N2-scan.txt', 'w') as fout:
    fout.write('     UHF 1.5->3.0     FCI      UHF 3.0->1.5     FCI\n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f  %12.8f\n'
                   % (xi, ehf1[i], emc1[i], ehf2[i], emc2[i]))


import matplotlib.pyplot as plt
plt.plot(x, ehf1, label='UHF,1.5->3.0')
plt.plot(x, ehf2, label='UHF,3.0->1.5')
plt.plot(x, emc1, label='FCI,1.5->3.0')
plt.plot(x, emc2, label='FCI,3.0->1.5')
plt.legend()
plt.show()