"""
 Custom sr-PBE functional based on Goll's paper: Chemical Physics 329 (2006) 276–282

 This customization in also based on the example 24-custom_xc_funcitonal.py from pySCF
"""
"""
Custom-wrapped PBE XC functional using define_xc_.

Analytic PBE-exchange implementation and a wrapper combining it with libxc PBE-correlation.
- Shows how to compute exc, (vrho, vgamma), and fxc second derivatives for GGA exchange.
- Uses libxc for correlation (so overall result matches builtin 'PBE' within integration tolerances).
- Also demonstrates how to handle spin (collinear) by using spin-scaling for exchange
  while letting libxc handle correlation (libxc is robust for spin).
"""

import numpy as np
from pyscf import gto, dft

# ---------- PBE constants ----------
# Cx = 3/4 (3/pi)^{1/3}
Cx = 3.0/4.0 * (3.0/np.pi)**(1.0/3.0)

# PBE parameters
kappa = 0.804
mu = 0.2195149727645171   # standard PBE mu

# prefactor in s definition: s = sqrt(sigma) / ( 2 * kf * rho^{4/3} )
kf_pref = (3.0 * np.pi**2)**(1.0/3.0)   # k_F factor

# ---------- helper functions ----------
def Fx_and_derivatives(s2):
    """Return F_x(s) and dF/d(s^2), d2F/d(s^2)^2 for PBE in terms of s^2.
    We compute derivatives wrt s^2 because s^2 = sigma * const / rho^{8/3}.
    """
    # avoid negative or extremely small s2
    s2 = np.maximum(s2, 0.0)
    denom = 1.0 + mu * s2 / kappa
    Fx = 1.0 + kappa - kappa / denom
    # dF/d(s^2) = d/ds2 [ kappa/(1 + mu s2/kappa) ] = kappa * (- mu/kappa) / denom^2
    dF_ds2 = (kappa) * (mu / (kappa)) / (denom**2)  # positive
    # second derivative
    d2F_ds22 = -2.0 * (kappa) * (mu/(kappa))**2 / (denom**3)
    return Fx, dF_ds2, d2F_ds22

# ---------- analytic PBE exchange (unpolarized) ----------
def pbe_exchange_unpolarized(rho, dx, dy, dz, deriv=2):
    """
    Compute PBE exchange for *spin-unpolarized* inputs.
    rho, dx,dy,dz are numpy arrays on the integration grid shapes.
    Returns exc, vxc=(vrho, vgamma), fxc=(d2_rho2, d2_rhosigma, d2_sigma2)
    (kxc left as None).
    """

    # protect against zero density
    rho = np.asarray(rho)
    small_rho = 1e-20
    rho_safe = np.maximum(rho, small_rho)

    # sigma = |grad rho|^2
    sigma = dx*dx + dy*dy + dz*dz

    # s^2 = sigma / [ 4 * kf_pref^2 * rho^{8/3} ]
    rho_43 = rho_safe**(4.0/3.0)
    rho_83 = rho_safe**(8.0/3.0)
    denom_s2 = 4.0 * (kf_pref**2) * rho_83
    s2 = sigma / denom_s2

    # LDA exchange energy density (per vol): eps_x_LDA = - Cx * rho^{4/3}
    eps_x_lda = -Cx * rho_43

    # enhancement factor and derivatives: Fx(s), dF/d(s^2), d2F/d(s^2)^2
    Fx, dF_ds2, d2F_ds22 = Fx_and_derivatives(s2)

    # full exchange energy density: eps_x = eps_x_lda * Fx
    eps_x = eps_x_lda * Fx

    # derivatives:
    # we need vrho = d eps_x / d rho   and vgamma = d eps_x / d sigma
    # compute partials step by step.

    # d eps_x_lda / d rho = - Cx * (4/3) rho^{1/3}
    d_epslda_drho = -Cx * (4.0/3.0) * (rho_safe**(1.0/3.0))

    # d s2 / d rho  and  d s2 / d sigma
    # s2 = sigma / denom_s2, denom_s2 = 4 kf^2 rho^{8/3}
    # d s2 / d sigma = 1 / denom_s2
    ds2_dsigma = 1.0 / denom_s2
    # d s2 / d rho = - sigma * d(denom_s2)/d rho / denom_s2^2
    # d denom_s2 / d rho = 4 kf^2 * (8/3) rho^{5/3} = 4*kf^2*(8/3)*rho^{5/3}
    ddenom_drho = 4.0 * (kf_pref**2) * (8.0/3.0) * (rho_safe**(5.0/3.0))
    ds2_drho = - sigma * ddenom_drho / (denom_s2**2)

    # dFx/d rho via chain: dFx/d rho = dF/d(s^2) * ds2/drho
    dFx_drho = dF_ds2 * ds2_drho
    # dFx/d sigma = dF/d(s^2) * ds2/dsigma
    dFx_dsigma = dF_ds2 * ds2_dsigma

    # assemble vrho:
    # vrho = d_epslda_drho * Fx + eps_x_lda * dFx_drho
    vrho = d_epslda_drho * Fx + eps_x_lda * dFx_drho

    # vgamma (derivative wrt sigma): vgamma = d eps_x / d sigma = eps_x_lda * dFx/dsigma
    vgamma = eps_x_lda * dFx_dsigma

    # second derivatives (fxc): d2 eps / d rho^2, d2 eps / d rho d sigma, d2 eps / d sigma^2
    # We compute:
    # d2 eps / d rho^2 = d^2 eps_lda/drho^2 * Fx + 2 * d_epslda_drho * dFx_drho + eps_x_lda * d2Fx/drho^2
    # but d2Fx/drho^2 = d2F/d(s^2)^2 * (ds2/drho)^2 + dF/d(s^2) * d2s2/drho^2
    # We'll compute only the dominant terms and a correct exact expression for d2s2/drho^2
    # (this is straightforward algebra but careful). We include all terms below.

    # d2 eps_lda/drho^2 = - Cx * (4/3)*(1/3) rho^{-2/3} = - Cx * (4/9) rho^{-2/3}
    d2_epslda_drho2 = -Cx * (4.0/9.0) * (rho_safe**(-2.0/3.0))

    # Now find d2 s2 / d rho^2 and d2 s2 / (d rho d sigma) and d2 s2 / d sigma^2
    # s2 = sigma / denom_s2  with denom_s2 = A * rho^{8/3}, A = 4 kf^2
    A = 4.0 * (kf_pref**2)
    # denom_s2 = A * rho^{8/3}
    # ds2/drho computed above; now d2s2/drho2:
    # ds2/drho = - sigma * A * (8/3) rho^{5/3} / (A^2 rho^{16/3}) = - sigma * (8/3) / (A rho^{11/3})
    # but to avoid algebra mistakes, we compute with safe numerical expressions:
    # Use symbolic-like finite expressions:
    # ds2_drho = - sigma * d(denom)/drho / denom^2   (we have these already)
    # d2s2_drho2 = - [ d sigma / drho * ddenom/drho + sigma * d2denom/drho2 ] / denom^2
    #              + 2 * sigma * (ddenom/drho)^2 / denom^3
    # For practical grids, d sigma / drho = 0 (sigma and rho are independent variables on the grid).
    # So the first term simplifies because d sigma / drho = 0. We then compute:
    # d2s2_drho2 = - sigma * d2denom_drho2 / denom^2 + 2 * sigma * (ddenom/drho)^2 / denom^3
    # compute d2denom/drho2:
    d2denom_drho2 = 4.0 * (kf_pref**2) * (8.0/3.0) * (5.0/3.0) * (rho_safe**(2.0/3.0))
    d2s2_drho2 = - sigma * d2denom_drho2 / (denom_s2**2) + 2.0 * sigma * (ddenom_drho**2) / (denom_s2**3)

    # d2s2 / (d rho d sigma) = d(1/denom_s2)/drho = - ddenom_drho / denom_s2^2
    d2s2_drho_dsigma = - ddenom_drho / (denom_s2**2)
    # d2s2 / d sigma^2 = 0

    # Now d2Fx/drho2:
    # d2Fx/drho2 = d2F/d(s^2)^2 * (ds2/drho)^2 + dF/d(s^2) * d2s2_drho2
    d2Fx_drho2 = d2F_ds22 * (ds2_drho**2) + dF_ds2 * d2s2_drho2

    # d2Fx / d rho d sigma = d2F/d(s^2)^2 * ds2_drho * ds2_dsigma + dF/d(s^2) * d2s2_drho_dsigma
    d2Fx_drho_dsigma = d2F_ds22 * ds2_drho * ds2_dsigma + dF_ds2 * d2s2_drho_dsigma

    # d2Fx / d sigma^2 = d2F/d(s^2)^2 * (ds2_dsigma)**2 + dF/d(s^2) * 0
    d2Fx_dsigma2 = d2F_ds22 * (ds2_dsigma**2)

    # assemble second derivatives:
    d2eps_drho2 = d2_epslda_drho2 * Fx \
                  + 2.0 * d_epslda_drho * dFx_drho \
                  + eps_x_lda * d2Fx_drho2

    d2eps_drho_dsigma = d_epslda_drho * dFx_dsigma + eps_x_lda * d2Fx_drho_dsigma

    d2eps_dsigma2 = eps_x_lda * d2Fx_dsigma2

    # pack outputs
    exc = eps_x  # energy density per point (no grid weight here; PySCF multiplies by weight)
    vxc = (vrho, vgamma)
    fxc = (d2eps_drho2, d2eps_drho_dsigma, d2eps_dsigma2)
    kxc = None

    if deriv < 2:
        # If only first derivatives requested, return exc and vxc with fxc None
        return exc, vxc, None, None
    return exc, vxc, fxc, kxc


def eval_pbe_custom(xc_code, rho, spin=0, relativity=0, deriv=2, omega=None, verbose=None):
    """
    Wrapper that provides analytic PBE exchange and uses libxc for correlation only.
    IMPORTANT: calls dft.libxc.eval_xc('0,PBE', ...) so libxc returns *correlation only*.
    """
    # Unpolarized (spin == 0): rho is (rho, dx,dy,dz)
    if spin == 0:
        rho0, dx, dy, dz = rho
        exc_x, vxc_x, fxc_x, kxc_x = pbe_exchange_unpolarized(rho0, dx, dy, dz, deriv=deriv)

        # CORRELATION-ONLY from libxc (no exchange)
        corr = dft.libxc.eval_xc(',PBE', rho, spin, relativity, deriv, verbose)
        exc_corr, vxc_corr, fxc_corr, kxc_corr = corr[0], corr[1], corr[2], corr[3]

        exc = exc_x + exc_corr
        vrho_total = vxc_x[0] + vxc_corr[0]
        vgamma_total = vxc_x[1] + vxc_corr[1]
        vxc = (vrho_total, vgamma_total)

        if (fxc_x is None) or (fxc_corr is None):
            fxc = None
        else:
            fxc = (fxc_x[0] + fxc_corr[0],
                   fxc_x[1] + fxc_corr[1],
                   fxc_x[2] + fxc_corr[2])
        return exc, vxc, fxc, None

    # Spin-polarized (collinear)
    if spin == 1:
        # Unpack spin densities & grads from PySCF spin GGA ordering
        rho_a, rho_b = rho[0], rho[1]
        dx_a, dy_a, dz_a = rho[2], rho[3], rho[4]
        dx_b, dy_b, dz_b = rho[5], rho[6], rho[7]

        rho_tot = rho_a + rho_b
        dx = dx_a + dx_b
        dy = dy_a + dy_b
        dz = dz_a + dz_b

        small = 1e-20
        rho_tot_safe = np.maximum(rho_tot, small)

        # spin scaling phi(zeta)
        zeta = (rho_a - rho_b) / rho_tot_safe
        zeta = np.clip(zeta, -1.0, 1.0)
        one_plus = np.maximum(1.0 + zeta, 1e-20)
        one_minus = np.maximum(1.0 - zeta, 1e-20)
        phi = 0.5 * (one_plus**(4.0/3.0) + one_minus**(4.0/3.0))

        # Compute analytic exchange for TOTAL density/gradient
        exc0, vxc0, fxc0, _ = pbe_exchange_unpolarized(rho_tot_safe, dx, dy, dz, deriv=deriv)
        eps0 = exc0  # (-Cx rho^{4/3})*Fx
        exc_x = eps0 * phi

        # propagate derivatives: vrho_tot and vgamma_tot
        vrho_eps0, vgamma_eps0 = vxc0
        vrho_tot = phi * vrho_eps0
        vgamma_tot = phi * vgamma_eps0

        # derivative due to spin-scaling phi(zeta)
        dphi_dz = (2.0/3.0) * (one_plus**(1.0/3.0) - one_minus**(1.0/3.0))
        deps_dz = eps0 * dphi_dz

        denom_t2 = rho_tot_safe**2
        dz_dra = 2.0 * rho_b / denom_t2
        dz_drb = -2.0 * rho_a / denom_t2

        vrho_a = vrho_tot + deps_dz * dz_dra
        vrho_b = vrho_tot + deps_dz * dz_drb

        # share gradient part equally (exchange depends on total sigma)
        vgamma_a = vgamma_tot
        vgamma_b = vgamma_tot

        # Build exchange vxc as tuple-of-tuples matching libxc spin-return shape
        vxc_x = ((vrho_a, vrho_b), (vgamma_a, vgamma_b))

        # CORRELATION-ONLY from libxc for spin case
        corr = dft.libxc.eval_xc(',PBE', rho, spin, relativity, deriv, verbose)
        exc_corr = corr[0]
        vxc_corr = corr[1]

        # Sum exchange + correlation
        exc_total = exc_x + exc_corr

        # Add vxc components (alpha/beta)
        vrho_a_tot = vxc_x[0][0] + vxc_corr[0][0]
        vrho_b_tot = vxc_x[0][1] + vxc_corr[0][1]
        vgamma_a_tot = vxc_x[1][0] + vxc_corr[1][0]
        vgamma_b_tot = vxc_x[1][1] + vxc_corr[1][1]
        vxc_total = ((vrho_a_tot, vrho_b_tot), (vgamma_a_tot, vgamma_b_tot))

        # skip fxc for spin-polarized analytic exchange (lengthy)
        return exc_total, vxc_total, None, None

    raise RuntimeError("Unsupported spin flag: {}".format(spin))

# Run after mf_ref.kernel() and mf_custom.kernel()
def energy_breakdown(mf):
    # returns dict with components
    e_tot = float(mf.e_tot)
    e_nuc = float(mf.energy_nuc())
    # energy_elec() returns (e_tot_elec, e1e + e2e?); use intrinsic helpers
    e_elec = mf.energy_elec()[0] if hasattr(mf, 'energy_elec') else None
    # we can compute coulomb (J) and exchange-correlation via nr_vxc integration
    ni = mf._numint
    grids = mf.grids

    ao = ni.eval_ao(mf.mol, grids.coords, deriv=1)
    dm = mf.make_rdm1()
    rho = ni.eval_rho(mf.mol, ao, dm, xctype='GGA')
    # libxc full xc (for reference) and coulomb via mean-field potential integrals
    exc_full = dft.libxc.eval_xc(mf.xc if isinstance(mf.xc, str) else 'PBE,PBE', rho, mol.spin, 0, 0, verbose=None)[0]
    E_xc = float(np.dot(exc_full, grids.weights))
    # compute Coulomb energy via Hartree:
    # build J (Coulomb) as 0.5 * sum_{pq,rs} DM * (pq|rs) * DM  -- use mf.get_j() or mf._coulG?
    # Simpler: use mf.get_j(mol, dm) from scf.routines
    try:
        jmat = mf.get_j(mf.mol, dm)
        # Coulomb energy = 0.5 * trace(dm @ jmat)
        E_coul = 0.5 * np.einsum('ij,ji', dm, jmat).real
    except Exception:
        E_coul = None

    return dict(e_tot=e_tot, e_nuc=e_nuc, e_elec=e_elec, E_coul=E_coul, E_xc=E_xc)


# ---------- Example usage and validation ----------
if __name__ == '__main__':
    mol = gto.M(
        atom = '''
        O  0.   0.       0.
        H  0.   -0.757   0.587
        H  0.   0.757    0.587 ''',
        basis = 'ccpvdz')

    # Print a short fingerprint to confirm the molecule
    print("Molecule summary:")
    print("  atoms:", mol.atom)  # small debug; mol.atom gives list/tuple
    print("  basis:", mol.basis)
    print("  nelectron:", mol.nelectron)
    print()

    # --- reference builtin PBE calculation ---
    mf_ref = dft.RKS(mol)
    mf_ref.xc = 'PBE'
    mf_ref.verbose = 3
    # optionally set grid level to ensure stable comparison
    mf_ref.grids.level = 4

    # run and immediately store the energy in a dedicated name E_ref
    _ = mf_ref.kernel()
    E_ref = float(mf_ref.e_tot)  # store as plain Python float to avoid accidental overwrite
    print("Reference (builtin PBE) E_ref = {:.12f} Ha".format(E_ref))
    print()


    # --- make sure we do not accidentally modify 'mol' or mf_ref later ---
    # (Now call the custom functional run using the same 'mol' object.)

    # --- custom run using your eval_pbe_custom registered via define_xc_ ---
    # (Assume eval_pbe_custom is in scope in this session; if not, import/define it first)
    mf_custom = dft.RKS(mol)
    # copy grids object to ensure identical integration settings
    mf_custom.grids = mf_ref.grids
    # register the custom xc (replace eval_pbe_custom with your callable name)
    
    mf_custom = mf_custom.define_xc_(eval_pbe_custom, 'GGA')  # <- ensure eval_pbe_custom exists
    mf_custom.verbose = 3

    _ = mf_custom.kernel()

    ni = mf_custom._numint
    grids = mf_custom.grids

    # Build AO and rho on the grid robustly
    ao = ni.eval_ao(mf_custom.mol, grids.coords, deriv=1)
    dm = mf_custom.make_rdm1()
    rho_grid = ni.eval_rho(mf_custom.mol, ao, dm, xctype='GGA')  # (rho, dx,dy,dz)

    # Basic normalization check
    rho0 = rho_grid[0]
    N_grid = np.dot(rho0, grids.weights)
    print("Normalization check: ∫ rho d r  = {: .12f}  (mol.nelectron = {})".format(N_grid, mf_custom.mol.nelectron))

    # 1) Compare LDA-exchange (libxc) vs analytic LDA term computed from your routine
    # libxc LDA exchange only: call 'LDA_X' or PBE exchange-only via 'PBE,0'
    # (libxc names vary; 'LDA_X' is standard for pure LDA exchange)
    try:
        lda_x_libxc = dft.libxc.eval_xc('LDA_X,', rho_grid, 0, 0, 0, verbose=None)[0]
        E_lda_libxc = np.dot(lda_x_libxc, grids.weights)
    except Exception:
        # fallback: use PBE exchange-only from libxc ('PBE,0')
        lda_x_libxc = dft.libxc.eval_xc('PBE,', rho_grid, 0, 0, 0, verbose=None)[0]
        E_lda_libxc = np.dot(lda_x_libxc, grids.weights)

    # analytic LDA from our code: eps_x_lda = -Cx * rho^{4/3}
    Cx = 3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3.0)
    eps_x_lda_analytic = - Cx * (rho0 ** (4.0 / 3.0))
    E_lda_analytic = np.dot(eps_x_lda_analytic, grids.weights)

    print("LDA-exchange (libxc) integral = {: .12f} Ha".format(E_lda_libxc))
    print("LDA-exchange (analytic) integral = {: .12f} Ha".format(E_lda_analytic))
    if abs(E_lda_libxc) > 1e-12:
        print(" ratio libxc/analytic = {: .6f}".format(E_lda_libxc / E_lda_analytic))

    # 2) Compare full exchange: analytic-PBE vs libxc PBE-exchange-only
    # analytic full PBE-exchange (what your routine produces)
    exc_x_arr, _, _, _ = pbe_exchange_unpolarized(rho0, rho_grid[1], rho_grid[2], rho_grid[3], deriv=0)
    E_x_analytic = np.dot(exc_x_arr, grids.weights)

    # libxc PBE exchange-only
    pbe_x_libxc = dft.libxc.eval_xc('PBE,', rho_grid, 0, 0, 0, verbose=None)[0]
    E_x_libxc = np.dot(pbe_x_libxc, grids.weights)

    print("PBE-exchange (analytic) integral = {: .12f} Ha".format(E_x_analytic))
    print("PBE-exchange (libxc PBE,0) integral = {: .12f} Ha".format(E_x_libxc))
    if abs(E_x_libxc) > 1e-12:
        print(" ratio libxc/analytic = {: .6f}".format(E_x_libxc / E_x_analytic))

    # 3) Full PBE XC from libxc (for reference)
    exc_libxc_full = dft.libxc.eval_xc('PBE,PBE', rho_grid, 0, 0, 0, verbose=None)[0]
    E_xc_libxc = np.dot(exc_libxc_full, grids.weights)
    print("PBE total (libxc 'PBE,PBE') = {: .12f} Ha".format(E_xc_libxc))

    # 4) Print a few sample grid points (first 5) for rho, eps_x_lda_analytic, and exc_x_arr
    nprint = min(5, rho0.size)
    print("\nSample grid values (first {} points):".format(nprint))
    for i in range(nprint):
        print(
            " i={:2d}  rho={: .6e}  eps_lda(analytic)={: .6e}  eps_x_analytic={: .6e}  eps_x_libxc_pbe={: .6e}".format(
                i, float(rho0[i]), float(eps_x_lda_analytic[i]), float(exc_x_arr[i]), float(pbe_x_libxc[i])
            ))

    # 5) Print totals from earlier run (for cross-check)
    print("\nmf_custom.e_tot = {: .12f} Ha".format(mf_custom.e_tot))
    print("mf_ref.e_tot    = {: .12f} Ha".format(E_ref))

    # assume rho0, exc_x_arr, pbe_x_libxc (arrays) and grids already available from previous diagnostic
    # If not in scope, re-evaluate them as in earlier code blocks.

    eps_lda = eps_x_lda_analytic  # your analytic LDA array from previous block
    eps_lib = pbe_x_libxc  # libxc's PBE-exchange-only per-grid array

    # avoid division by tiny numbers: only consider points where |eps_lda| > small
    small = 1e-30
    mask = np.abs(eps_lda) > small
    ratios = np.full(eps_lda.shape, np.nan)
    ratios[mask] = eps_lib[mask] / eps_lda[mask]

    # print robust statistics of the ratio
    print("ratio stats for eps_lib / eps_lda (only points with |eps_lda|>{}):".format(small))
    print("  median = ", np.nanmedian(ratios))
    print("  mean   = ", np.nanmean(ratios))
    print("  std    = ", np.nanstd(ratios))
    print("  min    = ", np.nanmin(ratios))
    print("  max    = ", np.nanmax(ratios))

    # print first 20 ratio values (for visual inspection)
    print("\nsample ratios (first 20 points, NaN means eps_lda too small):")
    print(ratios[:20])

    # Build spin-decomposed trial LDA exchange assuming closed-shell split rho/2
    rho_total = rho0
    rho_half = rho_total * 0.5
    eps_lda_spin_decomposed = - Cx * (rho_half ** (4.0 / 3.0) + rho_half ** (4.0 / 3.0))  # = -Cx * 2*(rho/2)^{4/3}
    E_lda_spin_decomp = np.dot(eps_lda_spin_decomposed, grids.weights)
    print("E_lda_spin_decomposed (using rho/2 per spin) = ", E_lda_spin_decomp)
    print("E_lda_libxc = ", E_lda_libxc)
    print("ratio libxc / spin-decomp = ", E_lda_libxc / E_lda_spin_decomp)

    # find a grid point with significant density
    idx = np.argmax(rho0)  # index of max density on grid (likely near nuclei)
    print("largest-rho index", idx)
    print(" rho[idx] = ", rho0[idx])
    print(" eps_lda_analytic[idx] = ", eps_lda[idx])
    print(" eps_lda_libxc(idx)   = ", pbe_x_libxc[idx])  # or call appropriate libxc arrays
    print(" grids.weights[idx]    = ", grids.weights[idx])


    ## ENERGY DECOMPOSITION:
    ref_break = energy_breakdown(mf_ref)
    cust_break = energy_breakdown(mf_custom)

    print("Reference breakdown (builtin PBE):")
    for k, v in ref_break.items():
        print(f"  {k:8s} = {v: .12f}")
    print("\nCustom breakdown (analytic-exchange + libxc-corr):")
    for k, v in cust_break.items():
        print(f"  {k:8s} = {v: .12f}")

    print("\nDifference (custom - ref):")
    for k in ref_break:
        if ref_break[k] is None or cust_break[k] is None:
            print(f"  {k:8s} = (one side None)")
        else:
            print(f"  {k:8s} = {cust_break[k] - ref_break[k]: .12e}")

    exit()
    # Run *after* mf_custom.kernel() has converged.
    # mf_custom should be the KS object registered with eval_pbe_custom.

    ni = mf_custom._numint
    grids = mf_custom.grids

    # 1) Evaluate AO (and derivatives) at grid points in the API shape numint expects.
    # Use deriv=1 so AO derivatives are provided (GGA needs grads).
    # Different pyscf versions may require different arg names; deriv=1 is widely supported.
    ao = ni.eval_ao(mf_custom.mol, grids.coords, deriv=1)

    # Quick debug: print the AO object type and structure so we can see what we got
    print("DEBUG: type(ao) =", type(ao))
    # If ao is a tuple/list, show shapes of elements
    if isinstance(ao, (list, tuple)):
        for i, a in enumerate(ao):
            try:
                print(f" DEBUG: ao[{i}].shape = {a.shape}")
            except Exception as e:
                print(f" DEBUG: ao[{i}] -- not an array: {type(a)}, repr: {repr(a)[:200]}")

    # 2) Build density matrix (RKS or UKS)
    dm = mf_custom.make_rdm1()
    print("DEBUG: dm.shape =", np.shape(dm))

    # 3) Evaluate rho on the grid. Pass xctype='GGA' so we get (rho, dx,dy,dz) (or spin variant)
    try:
        rho_grid = ni.eval_rho(mf_custom.mol, ao, dm, xctype='GGA')
    except Exception as exc:
        print("ERROR: ni.eval_rho raised:", exc)
        print(" -> show AO and DM details to debug further; try passing ao[0] if ao is tuple of arrays.")
        # try the common fallback: ao may be a tuple where ao[0] is the actual AO array
        if isinstance(ao, (list, tuple)) and len(ao) > 0:
            try:
                print("Trying fallback: ni.eval_rho(mol, ao[0], dm, xctype='GGA') ...")
                rho_grid = ni.eval_rho(mf_custom.mol, ao[0], dm, xctype='GGA')
                print("Fallback succeeded.")
            except Exception as exc2:
                print("Fallback also failed:", exc2)
                raise
        else:
            raise

    # analytic exchange (unpolarized or spin-aware)
    if mol.spin == 0:
        # rho_grid shape: (rho, dx,dy,dz)
        exc_x_arr, _, _, _ = pbe_exchange_unpolarized(rho_grid[0], rho_grid[1],
                                                      rho_grid[2], rho_grid[3], deriv=0)
        # correlation-only from libxc
        exc_corr_arr = dft.libxc.eval_xc(',PBE', rho_grid, 0, 0, 0, verbose=None)[0]

        # integrate over grid: energy = sum(exc * weight)
        E_x = np.dot(exc_x_arr, grids.weights)
        E_c = np.dot(exc_corr_arr, grids.weights)
        E_xc_sum = E_x + E_c

        # total XC from libxc('PBE,PBE') for reference (should equal builtin PBE xc)
        exc_libxc_full = dft.libxc.eval_xc('PBE,PBE', rho_grid, 0, 0, 0, verbose=None)[0]
        E_xc_libxc = np.dot(exc_libxc_full, grids.weights)

        print("Grid-integrated components (spin=0):")
        print("  E_x (analytic)         = {: .12f} Ha".format(E_x))
        print("  E_c (libxc, '0,PBE')   = {: .12f} Ha".format(E_c))
        print("  E_x + E_c              = {: .12f} Ha".format(E_xc_sum))
        print("  E_xc (libxc 'PBE,PBE') = {: .12f} Ha".format(E_xc_libxc))
        print("  mf_custom.e_tot        = {: .12f} Ha".format(mf_custom.e_tot))
        # If E_x + E_c ≈ E_xc_libxc, then our analytic exchange matches libxc exchange.
    else:
        # spin==1: evaluate analytic-exchange via the wrapper (eval_pbe_custom) on spin rho_grid
        exc_xc_from_wrapper = eval_pbe_custom('MY_PBE', rho_grid, spin=1, deriv=0)[0]
        # exc_xc_from_wrapper contains exc_x + exc_corr with our wrapper's logic
        E_xc_wrapper = np.dot(exc_xc_from_wrapper, grids.weights)

        # correlation-only from libxc
        exc_corr_arr = dft.libxc.eval_xc('0,PBE', rho_grid, 1, 0, 0, verbose=None)[0]
        E_c = np.dot(exc_corr_arr, grids.weights)

        # compute analytic-exchange by difference (wrapper - corr)
        # (only safe if the wrapper is our eval_pbe_custom)
        E_x = E_xc_wrapper - E_c

        # libxc full reference
        exc_libxc_full = dft.libxc.eval_xc('PBE,PBE', rho_grid, 1, 0, 0, verbose=None)[0]
        E_xc_libxc = np.dot(exc_libxc_full, grids.weights)

        print("Grid-integrated components (spin=1):")
        print("  E_x (analytic, by difference) = {: .12f} Ha".format(E_x))
        print("  E_c (libxc, '0,PBE')         = {: .12f} Ha".format(E_c))
        print("  E_x + E_c                    = {: .12f} Ha".format(E_x + E_c))
        print("  E_xc (libxc 'PBE,PBE')       = {: .12f} Ha".format(E_xc_libxc))
        print("  mf_custom.e_tot              = {: .12f} Ha".format(mf_custom.e_tot))

exit()
def b(mu):
    b_pbe = 0.21951
    alpha_x = 19.0 # damping off the gradient term (Physical Chemistry Chemical Physics (2005), 7(23), 3917-3923.)
    b_res = b_pbe * np.exp*(-alpha_x * mu**2)
    return b_res

def F_x(mu, s):
    kappa = 0.840

    #= = Compute b(mu)
    b_mu = b(mu)

    #= = compute Fx
    denom = b_mu * (s**2) /kappa
    denom = 1 + denom
    Fx = 1 + kappa
    Fx += (-kappa/denom)
    return Fx


def eval_srpbe_xc(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
    """
    Defining sr-PBE functional

    Parameters are the same as libxc.eval_xc:
      - rho: tuple of arrays in the ordering PySCF uses for GGA: (rho, dx, dy, dz)
      - deriv: requested derivative order (1 -> return vxc; 2 -> return fxc if available)
    Returns: (exc, vxc, fxc, kxc) matching libxc convention
    """

    #= = = Initial definitions
    rho0, dx, dy, dz = rho
    gamma = (dx ** 2 + dy ** 2 + dz ** 2)

    #kf = math.cbrt(3 * np.pi**2 * rho)


    # call libxc's implementation directly and return it unchanged
    # note: for GGA we pass 'PBE,PBE' meaning exchange=PBE, correlation=PBE
    return dft.libxc.eval_xc('PBE,PBE', rho, spin, relativity, deriv, verbose)