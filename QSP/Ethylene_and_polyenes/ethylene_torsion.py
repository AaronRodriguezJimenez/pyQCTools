"""
  Here we compute the Potential energy surface for the lowest energy singlet and triplet states
  for C2H4 along the torsion of CH2 and get the AS Hamiltonians for QKSD testing.
"""
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
import os

def ethylene():
    """
    Ethylene molecule from
    Chem. Phys. Lett., 248 (1996) 336
    """
    return [
        ['C', [-0.665350,  0.000000, 0.0000000]],
        ['H', [-1.229148,  0.922499, 0.0000000]],
        ['H', [-1.229148, -0.922499, 0.0000000]],
        ['C', [ 0.665350,  0.000000, 0.0000000]],
        ['H', [ 1.229148,  0.922499, 0.0000000]],
        ['H', [ 1.229148, -0.922499, 0.0000000]]
    ]

R_CC = 1.3307
R_CH = 1.0809
CCH_deg = 121.44

def build_ethene_from_internal(Rcc=R_CC, Rch=R_CH, cch_deg=CCH_deg):
    """
    Build planar ethylene (C2H4) coordinates using:
      - C1 at (-Rcc/2, 0, 0), C2 at (+Rcc/2, 0, 0)
      - Each C has two H atoms in the xy-plane, symmetric about the x-axis
      - angle C-C-H is the angle between the C->otherC vector (+x) and the C->H bond
    Returns list of (symbol, np.array([x,y,z])) with 0-based ordering:
      0:C1, 1:H_a1, 2:H_b1, 3:C2, 4:H_a2, 5:H_b2
    """
    theta = np.deg2rad(cch_deg)   # C-C-H angle in radians
    #theta = cch_deg  # C-C-H angle in radians

    # Carbon positions
    c1 = np.array([-Rcc/2.0, 0.0, 0.0])
    c2 = np.array([ Rcc/2.0, 0.0, 0.0])

    # For carbon 1, the vector toward the other carbon is +x, so C1->H makes angle theta from +x.
    # Place two H symmetrically above/below the x-axis (±y).
    h1_rel = np.array([Rch * np.cos(theta),  Rch * np.sin(theta), 0.0])
    h2_rel = np.array([Rch * np.cos(theta), -Rch * np.sin(theta), 0.0])
    H1 = c1 + h1_rel
    H2 = c1 + h2_rel

    # For carbon 2, the local "C->otherC" direction is toward -x; to keep the same angle C-C-H
    # we measure the angle relative to the direction from C2 to C1 (which is -x).
    # Equivalent construction: use angle (pi - theta) from +x, giving cos(pi-theta) = -cos(theta),
    # sin(pi-theta) = sin(theta). So relative vectors will be mirrored in x sign.
    h3_rel = np.array([ -Rch * np.cos(theta),  Rch * np.sin(theta), 0.0])
    h4_rel = np.array([ -Rch * np.cos(theta), -Rch * np.sin(theta), 0.0])
    H3 = c2 + h3_rel
    H4 = c2 + h4_rel

    mol = [
        ('C', c1.copy()),
        ('H', H1.copy()),
        ('H', H2.copy()),
        ('C', c2.copy()),
        ('H', H3.copy()),
        ('H', H4.copy())
    ]
    return mol

def rodrigues_rotate_point(p, axis_point, axis_dir, angle_rad):
    """
    Rotate point p around an axis defined by (axis_point, axis_dir).
    axis_dir must be non-zero; angle_rad is in radians.
    Uses Rodrigues' rotation formula.
    Returns rotated 3-vector.
    """
    k = np.asarray(axis_dir, dtype=float)
    k_norm = np.linalg.norm(k)
    if k_norm == 0.0:
        raise ValueError("axis_dir must be non-zero")
    k = k / k_norm

    r = np.asarray(p, dtype=float) - np.asarray(axis_point, dtype=float)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Rodrigues: r_rot = r*cos + (k x r)*sin + k*(k·r)*(1-cos)
    r_rot = r * cos_a + np.cross(k, r) * sin_a + k * (np.dot(k, r) * (1.0 - cos_a))
    return axis_point + r_rot

def torsion_scan(molecule, fragment_indices_to_rotate, cc_idx1, cc_idx2, angles_deg):
    """
    Build a list of full-molecule geometries for each torsion angle.
    angles_deg: iterable of angles in degrees
    Returns list of molecules (each is list like molecule).
    """
    geoms = []
    for ang in angles_deg:
        geoms.append(rotate_fragment_about_cc(molecule, fragment_indices_to_rotate, cc_idx1, cc_idx2, ang))
    return geoms

def rotate_fragment_about_cc(mol_coords, rotate_indices, cc_idx1=0, cc_idx2=3, angle_deg=0.0):
    """Return new deep-copied molecule coords with given atom indices rotated about CC axis (cc_idx1->cc_idx2)."""
    mol_copy = [[sym, coord.copy()] for sym, coord in mol_coords]
    p1 = mol_copy[cc_idx1][1]
    p2 = mol_copy[cc_idx2][1]
    axis = p2 - p1
    angle_rad = np.deg2rad(angle_deg)
    for i in rotate_indices:
        mol_copy[i][1] = rodrigues_rotate_point(mol_copy[i][1], axis_point=p1, axis_dir=axis, angle_rad=angle_rad)
    return mol_copy

def mol_to_pyscf_atom_string(mol_coords):
    """Convert [(sym, np.array)] -> multiline atom string 'C x y z; H x y z; ...'"""
    parts = []
    for sym, coord in mol_coords:
        parts.append(f"{sym} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}")
    return '; '.join(parts)

# --- SCF wrappers ---

def run_singlet_energy(atom_string, basis, output, twist=True):
    mol = gto.M(atom=atom_string, basis=basis, unit='Angstrom', charge=0, spin=0)
    mol.spin = 0
    mol.verbose = 4
    mol.output = output
    if twist:
        mol.symmetry = 'D2'
    else:
        mol.symmetry = 'D2h'
    mol.build()
    mf = scf.RHF(mol)
    mf.max_cycle = 100
    mf.conv_tol = 1E-9
    e = mf.kernel()

    # - - Check SCF Stability
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
    # e is total electronic energy (Hartree)
    return ehf_val, mf, mol

def run_triplet_energy(atom_string, basis, output, twist=True):
    """Run with given spin (2 for triplet). spin is 2*S (number of unpaired electrons)."""
    mol = gto.M(atom=atom_string, basis=basis, unit='Angstrom', charge=0)
    mol.spin = 2
    mol.output = output
    if twist:
        mol.symmetry = 'D2'
    else:
        mol.symmetry = 'D2h'

    mol.build()
    mf = scf.UHF(mol)
    mf.max_cycle = 100
    mf.conv_tol = 1E-9
    e = mf.kernel()
    # - - Check SCF Stability
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
    # e is total electronic energy (Hartree)
    return ehf_val, mf

def run_cas(mf, dm, ncas_orbs, ncas_elec, ci=None, twist=True):

    mc = mcscf.CASCI(mf, ncas_orbs, ncas_elec)
    mc.fcisolver.conv_tol = 1e-7
    mc.fcisolver.threads = 1

    # === define your CAS(4,4) target (change irrep keys to match HOMO/LUMO labels) ===
    nactel = 4
    # For D2h symmetry
    if twist:
        # For D2 symmetry
        ncas = {
            'A': 1,  # sigma
            'B3': 1,  # pi
            'B2': 1,  # pi*
            'B1': 1,  # sigma*
        }

        ncore = {
            'A': 2,  # C 1s + C–H σ combinations
            'B1': 2,  # 1C-C sigma bonding + 2C-H σ
            'B2': 1,  # C-H σ
            'B3': 1,  # C-H σ
        }
    else:
        ncas = {
            'Ag':  1,  # sigma
            'B3u': 1,  # pi
            'B2g': 1,  # pi*
            'B1u': 1,  # sigma*
            }

        ncore = {
            'Ag':  2, # C 1s + C–H σ combinations
            'B1u': 2, # 1C-C sigma bonding + 2C-H σ
            'B2u': 1, #C-H σ
            'B3g': 1, #C-H σ
        }

    mo = mcscf.sort_mo_by_irrep(mc, mf.mo_coeff, ncas, ncore)
    e_tot, _, ci, mo, _ = mc.kernel(mo, ci)
    mc.analyze()

    return mf.make_rdm1(), mo, ci

def get_active_space_tensors(mol, mf, localized=False):
    """
    Robustly extracts active space tensors from SymAdaptedCASSCF or CASSCF.
    """
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
        loc_occ = lo.Boys(mol, C_act_occ).kernel(verbose=4)
        loc_vir = lo.Boys(mol, C_act_vir).kernel(verbose=4)

        C_loc = np.column_stack((loc_occ, loc_vir))
        mo_coeff_final = C_loc

    else:
        mo_coeff_final = C

    U = mo_coeff_final

    h1_spatial = U.T @ ao_obi @ U  # Rotated single-body integrals.
    h2_spatial = ao2mo.incore.full(ao_eri, U)  # Rotated two-body integrals.
    h0 = nuc
    return h0, h1_spatial, h2_spatial

#========================================================================================
# Calculation Begins:
E_RHF = []
E_cas_singlet = []
E_cas_triplet = []

# --- Example usage for ethylene:
mol =  build_ethene_from_internal()

# Index mapping (0-based)
# 0: C (left)
# 1: H (attached to left C)
# 2: H (attached to left C)
# 3: C (right)
# 4: H (attached to right C)
# 5: H (attached to right C)

# We will rotate the "right" CH2 fragment (atoms 3,4,5) about the C-C axis defined by atoms 0->3.
# Note: atom 3 (the right carbon) lies on the axis; rotating it around the axis leaves it unchanged.
fragment_to_rotate = [3, 4, 5]
cc_idx_left = 0
cc_idx_right = 3


print("Initial Geometry")
for sym, coord in mol:
    print(f"{sym:>2}  {coord[0]: .6f} {coord[1]: .6f} {coord[2]: .6f}")

angles = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0, 90.0, 95.0]
geometries = torsion_scan(mol, fragment_to_rotate, cc_idx_left, cc_idx_right, angles)
bas = "sto3g"

#- - - Tensor functions
def save_tensors(b, h0, h1, h2, folder="tensors"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Filename includes the bond distance for easy identification
    filename = os.path.join(folder, f"{b}_tensors.npz")

    # Save arrays; h0 is stored as a 0-d array
    np.savez_compressed(filename, hc=h0, h1e=h1, h2e=h2)
    print(f"Tensors saved to {filename}")

def create_save_H_active(folder, Hname, mol, mf):
    """
    folder :: string - A root for saving results in a folder
     Hname : string - Rotation angle
    :return:
    """

    # Simple extraction
    h0_spatial, h1_spatial, h2_spatial = get_active_space_tensors(mol, mf, localized=True)
    save_tensors(Hname, h0_spatial, h1_spatial, h2_spatial, folder)
    return


for i, geom in enumerate(geometries):
    if i == 0: # computing configuration at 0deg
        print(f"\nGeometry for 0° planar configuration")
        pyscf_mol = mol_to_pyscf_atom_string(geom)
        output = f"/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/C2H4_rotation/C2H4_singlet_{angles[i]:.1f}.out"
        e_singlet, mf_singlet, mol_singlet = run_singlet_energy(pyscf_mol, bas, output, twist=False)
        mc = mcscf.CASCI(mf_singlet, 4, 4)
        e_cas_sing, _, _, mo, _ = mc.kernel()

        E_RHF.append(e_singlet)
        E_cas_singlet.append(e_cas_sing)

        #- - - Extract and save Hamiltonian
        Hname = f"{angles[i]:.1f}"
        folder = "/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/C2H4_rotation"
        create_save_H_active(folder, Hname, mol_singlet, mf_singlet)
        mc.analyze()

        output = f"/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/C2H4_rotation/C2H4_triplet_{angles[i]:.1f}.out"
        e_triplet, mf_triplet = run_triplet_energy(pyscf_mol, bas, output, twist=False)
        mc = mcscf.CASCI(mf_triplet, 4, (3, 1))
        e_cas_trip, _, _, _, _ = mc.kernel(mo)
        mc.analyze()
        E_cas_triplet.append(e_cas_trip)

    else:
        print(f"\nGeometry for angle = {angles[i]:.1f}°")
        for sym, coord in geom:
                print(f"{sym:>2}  {coord[0]: .6f} {coord[1]: .6f} {coord[2]: .6f}")

        pyscf_mol = mol_to_pyscf_atom_string(geom)
        output = f"/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/C2H4_rotation/C2H4_singlet_{angles[i]:.1f}.out"
        e_singlet, mf_singlet, mol_singlet = run_singlet_energy(pyscf_mol, bas, output, twist=True)

        mc = mcscf.CASCI(mf_singlet, 4, 4)
        e_cas_sing, _, _, mo, _ = mc.kernel()
        mc.analyze()

        E_RHF.append(e_singlet)
        E_cas_singlet.append(e_cas_sing)

        # - - - Extract and save Hamiltonian
        Hname = f"{angles[i]:.1f}"
        folder = "/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/C2H4_rotation"
        create_save_H_active(folder, Hname, mol_singlet, mf_singlet)

        output = f"/Users/admin/PycharmProjects/pyQCTools/QSP/Ethylene_and_polyenes/C2H4_rotation/C2H4_triplet_{angles[i]:.1f}.out"
        e_triplet, mf_triplet = run_triplet_energy(pyscf_mol, bas, output, twist=True)
        mc = mcscf.CASCI(mf_triplet, 4, (3,1))
        e_cas_trip, _, _, _, _ = mc.kernel(mo)
        mc.analyze()

        E_cas_triplet.append(e_cas_trip)
#
# Plot Energy results
#
import matplotlib.pyplot as plt
plt.figure()
plt.plot(angles, E_RHF, label="RHF-singlet", color='b')
plt.plot(angles, E_cas_singlet, label="CAS-singlet", color='k')
plt.plot(angles, E_cas_triplet, label="CAS-triplet", color='k', linestyle='--')
plt.title("RHF energies")
plt.ylabel("Energy (a.u.)")
plt.xlabel("Torsion angle (deg)")
plt.legend()
plt.show()