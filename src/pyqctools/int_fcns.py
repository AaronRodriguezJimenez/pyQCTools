"""
  Feb 2026
  Collection of functions designed for the computation, selection, extraction and storage of
  molecular integrals (Chemist's notation is used here).
"""
import numpy as np
from pyscf import ao2mo

#- - - Storage functions
def save_restricted_integrals(b, h0, h1, h2, folder="tensors"):
    """ Save integrals from a restricted scf scheme.
    b - tittle
    h0, h1, h2 -> Nuclear, 1e-integrals, 2e-integrals
    """
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Filename includes the bond distance for easy identification
    filename = os.path.join(folder, f"{b}_integrals.npz")

    # Save arrays; h0 is stored as a 0-d array
    np.savez_compressed(filename, hc=h0, h1e=h1, h2e=h2)
    print(f"Integrals saved to {filename}")

def save_unrestricted_integrals(b, h0, h1a, h1b, eri_aa, eri_bb, eri_ab, folder="tensors"):
    """
    b - tittle (str)
    """
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Filename includes the bond distance for easy identification
    filename = os.path.join(folder, f"{b}_integrals.npz")

    # Save arrays; h0 is stored as a 0-d array
    np.savez_compressed(filename, hc=h0, h1_a=h1a, h1_b=h1b,
                        h2aa=eri_aa, h2bb=eri_bb, h2ab=eri_ab)
    print(f"Tensors saved to {filename}")

def get_restricted_ints_full(mol, mf, localized=False):
    """
    Get molecular integrals from a restricted scf scheme (mf).
    Do not supports ncore frozen orbitals (zero means full orbital space)
    """
    print("Getting restricted integrals... ")
    print(f"Localized orbitals: {localized}")

    nuc = mol.energy_nuc()  # Nuclear repulsion.
    ao_kin = mol.intor('int1e_kin')  # Electronic kinetic energy.
    ao_nuc = mol.intor('int1e_nuc')  # Nuclear-electronic attraction.
    ao_obi = ao_kin + ao_nuc  # Single-electron hamiltonian.
    ao_eri = mol.intor('int2e')  # Electonic repulsion interaction.
    C = mf.mo_coeff

    # 1. Handle Localization if requested
    if localized:
        from pyscf import lo
        nocc = int((mf.mo_occ > 0).sum())
        occ_idx_act = np.arange(0, nocc)
        vir_idx_act = np.arange(nocc, len(C))
        # Split MOs into occupied / virtual subsets
        C_act_occ = C[:, occ_idx_act]
        C_act_vir = C[:, vir_idx_act]

        # Boys localization
        loc_occ = lo.Boys(mol, C_act_occ).kernel()
        loc_vir = lo.Boys(mol, C_act_vir).kernel()

        C_loc = np.column_stack((loc_occ, loc_vir))
        mo_coeff_final = C_loc

    else:
        mo_coeff_final = C

    U = mo_coeff_final

    h1_spatial = U.T @ ao_obi @ U  # Rotated single-body integrals.
    h2_spatial = ao2mo.incore.full(ao_eri, U)  # Rotated two-body integrals.
    h0 = nuc
    return h0, h1_spatial, h2_spatial

def get_unrestricted_ints_full(mol, mf, localized=False, debug=True):
    """
    Robustly extract integrals supporting UHF.
    mo_coeff is a tupple (moalpha, mobeta)
    Do not Supports frozen core
    Returns:
      h0, h1_alpha, h1_beta, h2_aa, h2_bb, h2_ab
    Optionally also Ca_final, Cb_final when return_mos=True.

    Set debug=True to print shapes during execution for easier debugging.
    """
    print("Getting unrestricted integrals... ")
    print(f"Localized orbitals: {localized}")

    # AO & nuclear
    nuc = mol.energy_nuc()  # Nuclear repulsion.
    ao_kin = mol.intor('int1e_kin', hermi=1)
    ao_nuc = mol.intor('int1e_nuc', hermi=1)
    ao_obi = ao_kin + ao_nuc
    ao_eri = mol.intor('int2e')
    n_ao = ao_obi.shape[0]

    # get mo_coeff and mo_occ robustly
    mo_coeff = getattr(mf, 'mo_coeff', None)
    if mo_coeff is None:
        raise ValueError("mf.mo_coeff is None")

    Ca, Cb = mo_coeff.copy()

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

    # Localization (Boys) separately for occ/vir, per spin
    if localized:
        from pyscf import lo
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

    # Nuclear Repulsion Term
    h0 = nuc
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

def get_restricted_active_space_integrals(mol, mf, mc, localized=False, include_all_noncore=False):
    """
    Compute molecular integrals for a selected active space from a restricted SCF
    case. Extract effective active-space Hamiltonian for either:
      - the CASSCF-defined active space (mc.ncas), or
      - a different selected non-core block (all non-core orbitals if include_all_noncore=True).

    Therefore it uses a mc pyscf object.
    :return: H0, H1, H2
    """
    C = mc.mo_coeff.copy()
    ncore = int(mc.ncore)
    nmo = C.shape[1]

    # determine target active count
    if include_all_noncore:
        target_ncas = nmo - ncore
    else:
        target_ncas = int(mc.ncas)

    if target_ncas <= 0:
        raise ValueError("target_ncas must be > 0")

    # 1. Handle Localization if requested
    if localized:
        from pyscf import lo
        nocc = int((mf.mo_occ > 0).sum())
        occ_idx_act = np.arange(0, nocc)
        vir_idx_act = np.arange(nocc, len(C))
        # Split MOs into occupied / virtual subsets
        C_act_occ = C[:, occ_idx_act]
        C_act_vir = C[:, vir_idx_act]

        # Boys localization
        loc_occ = lo.Boys(mol, C_act_occ).kernel(verbose=4)
        loc_vir = lo.Boys(mol, C_act_vir).kernel(verbose=4)

        C_loc = np.column_stack((loc_occ, loc_vir))
        mo_coeff_final = C_loc

    else:
        mo_coeff_final = C

    # -----------------------
    # 1) One-electron (h1) and core energy: call h1e_for_cas via get_h1eff
    h1_spatial, h0 = mc.get_h1eff(mo_coeff_final, ncas=target_ncas, ncore=ncore)

    # -----------------------
    # 2) Two-electron: compute using ao2mo.full with the exact MO block we want.
    start = ncore
    stop = ncore + target_ncas
    # MO block for claculation: shape (nao, target_ncas)
    mo_for_mol = mo_coeff_final[:, start:stop].copy()

    # Warning: building a full (n,n,n,n) tensor scales as n^4 memory.
    # For n=26, that's ~26^4 ~ 456,976 elements (float64 ~ 3.7 MB) — actually fine,
    # but for larger n this quickly explodes. Still, ao2mo.full may use intermediate memory.
    try:
        # If underlying SCF has precomputed eri available (faster), use it
        if hasattr(mc._scf, "_eri") and getattr(mc._scf, "_eri", None) is not None:
            eri = ao2mo.full(mc._scf._eri, mo_for_mol, max_memory=mc.max_memory)
        else:
            eri = ao2mo.full(mc.mol, mo_for_mol, verbose=mc.verbose, max_memory=mc.max_memory)
    except Exception as e:
        raise RuntimeError("Failed to build two-electron integrals with ao2mo.full: " + str(e)) from e

    # ao2mo.full returns chemists' (ij|kl) ordering as a 4-index array (n,n,n,n).
    # Convert to physicists' (pq|rs) ordering used by OpenFermion: (i,j,k,l) -> transpose(0,2,3,1)
    eri = np.asarray(eri)
    if eri.ndim != 4 or eri.shape != (target_ncas,) * 4:
        # If ao2mo returned a packed/flattened format for some reason, try to handle it robustly:
        if eri.ndim == 1:
            n_guess = int(round(eri.size ** 0.25))
            if n_guess ** 4 == eri.size:
                eri = eri.reshape((n_guess,) * 4)
            else:
                raise RuntimeError(f"Unexpected ao2mo.full output shape {eri.shape} (size {eri.size})")
        elif eri.ndim == 2:
            # some variants might return m x m where m=n*(n+1)/2  (packed). Unpack:
            m = eri.shape[0]
            n_guess = int(round((np.sqrt(1 + 8 * m) - 1) / 2))
            if n_guess * (n_guess + 1) // 2 == m:
                eri = ao2mo.restore(1, eri, n_guess)
            else:
                raise RuntimeError(f"Unexpected 2D ao2mo.full output shape {eri.shape}")
        else:
            raise RuntimeError(f"Unexpected ao2mo.full output ndim={eri.ndim}, shape={eri.shape}")

    h2_spatial = eri  # .transpose(0, 2, 3, 1).copy() #Use that transpose for the physicists notation

    return h0, h1_spatial, h2_spatial

def get_unrestricted_active_space_integrals(mol, mc, localized=False, include_all_noncore=False):
    """
    Using include all_noncore returns all integrals belonging to the full MO space except
    the core terms, equivalent to a frozen core case.

    Compute molecular integrals for a selected active space from a restricted SCF
    case. Extract effective active-space Hamiltonian for either:
      - the CASSCF-defined active space (mc.ncas), or
      - a different selected non-core block (all non-core orbitals if include_all_noncore=True).

    Therefore it uses a mc pyscf object.
    :return: H0, H1, H2
    """
    Ca, Cb = mc.mo_coeff.copy()
    ncore = int(mc.ncore)
    n_ao = Ca.shape[0]
    nmo = Ca.shape[1]

    # determine target active count
    if include_all_noncore:
        target_ncas = nmo - ncore
    else:
        target_ncas = int(mc.ncas)

    if target_ncas <= 0:
        raise ValueError("target_ncas must be > 0")

    # 1. Handle Localization if requested
    # number of MOs
    nmo_a = Ca.shape[1]
    nmo_b = Cb.shape[1]

    # count occupied orbitals
    a_nocc = int((mf.mo_occ[0] > 0).sum())
    b_nocc = int((mf.mo_occ[1] > 0).sum())

    # Localization (Boys) separately for occ/vir, per spin
    if localized:
        from pyscf import lo
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
        raise ValueError(
            f"Final MO coeff matrices must have {n_ao} rows (AO basis). Got {Ca_final.shape}, {Cb_final.shape}")

    # -----------------------
    # 1) One-electron (h1) and core energy: call h1e_for_cas via get_h1eff
    h1_alpha, h0 = mc.get_h1eff(Ca_final, ncas=target_ncas, ncore=ncore)
    h1_beta, h0 = mc.get_h1eff(Cb_final, ncas=target_ncas, ncore=ncore)

    # 2) Two-electron: compute using ao2mo.full with the exact MO block we want.
    start = ncore
    stop = ncore + target_ncas
    # MO block for claculation: shape (nao, target_ncas)
    Ca_final = Ca_final[:, start:stop].copy()
    Cb_final = Cb_final[:, start:stop].copy()

    h2_aa = ao2mo.kernel(mol, (Ca_final, Ca_final, Ca_final, Ca_final))
    h2_aa = ao2mo.restore(1, h2_aa, target_ncas)

    h2_bb = ao2mo.kernel(mol, (Cb_final, Cb_final, Cb_final, Cb_final))
    h2_bb = ao2mo.restore(1, h2_bb, target_ncas)

    h2_ab = ao2mo.kernel(mol, (Ca_final, Ca_final, Cb_final, Cb_final))
    h2_ab = ao2mo.restore(1, h2_ab, target_ncas)

    return h0, h1_alpha, h1_beta, h2_aa, h2_bb, h2_ab


    # -----------------------

if __name__ == "__main__":
    from pyscf import gto, scf
    print("Test int_fcns")
    b = 1.1
    mol = gto.Mole()
    mol.atom = [
        ['N', (0.0, 0.0, -b / 2)],
        ['N', (0, 0, b / 2)]]

    mol.basis = 'sto-3g'
    # mol.symmetry = 'D2h' # D2h is more numerically robust than Dooh
    mol.spin = 0
    mol.build()

    # - - - Test get restricted type integrals
    mf = scf.RHF(mol)
    mf.max_cycle = 100
    mf.conv_tol = 1e-8
    mf.run()
    print("E(UHF) = ", mf.e_tot)
    print("E(UHF) = ", mf.e_tot)

    h0, h1, h2 = get_restricted_ints_full(mol, mf, localized=False)
    print(h0)
    print(h1.shape)
    print(h2.shape)

    # - - - Test active space integrals (restricted)
    from pyscf import mcscf

    mc = mcscf.CASCI(mf, 2, 2)
    e_casci, _, _, _, _ = mc.kernel()
    h0, h1, h2 = get_restricted_active_space_integrals(mol, mc, localized=False)
    print(h0)
    print(h1.shape)
    print(h2.shape)
    # save_restricted_integrals("N2test", h0, h1, h2, "/Users/admin/PycharmProjects/pyQCTools/DBF/")

    # - - - Test get unrestricted type integrals
    mf = scf.UHF(mol)
    mf.max_cycle = 100
    mf.conv_tol = 1e-8
    mf.run()
    ss = mf.spin_square()
    print("E(UHF) = ", mf.e_tot)
    print(f"Spin square <S^2>: {ss[0]}")

    h0, h1a, h1b, h2_aa, h2_bb, h2_ab = get_unrestricted_ints_full(mol, mf, localized=False)
    print(h0)
    print(h1a.shape)
    print(h2_aa.shape)

    # - - - Test active space integrals (unrestricted)
    h0, h1_alpha, h1_beta, h2aa, h2bb, h2ab = get_unrestricted_active_space_integrals(mol, mc, localized=True)
    print(h0)
    print(h1_alpha.shape)
    print(h2aa.shape)
    save_unrestricted_integrals("N2test", h0, h1_alpha, h1_beta, h2aa, h2bb, h2ab, "/Users/admin/PycharmProjects/pyQCTools/DBF/")