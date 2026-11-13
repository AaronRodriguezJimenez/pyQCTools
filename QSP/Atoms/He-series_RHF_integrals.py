"""
 This script generates the one- and two-electron integrals for the He-series atoms (He, Li+, Be2+, B3+, C4+, N5+, O6+, F7+, Ne8+)
  using the PySCF library.
  The integrals are saved in a .npz file for later use in quantum simulations.
"""
from pyscf import gto, scf, fci
from openfermion import QubitOperator
from src.pyqctools.int_fcns import get_restricted_ints_full, save_restricted_integrals
import os

"""
 Class for computing atomic energies using
 RHF
 KS-DFT
 FCI

 Also computes the Hamiltonian in qubit form for use in VQEmulti.
"""
class AtomSystem:
    def __init__(self, atoms, charge, spin=0, basis="sto-3g"):
        """
        :param atoms: List of tuples, e.g. [("O", (0,0,0)),]
        :param basis: string defining the basis set (default 'sto3g')
        :param charge: the charge of the atom
        :param spin: 2S where 0-for closed-shell singlets
        """
        self.atoms = atoms
        self.charge = charge
        self.spin = spin
        self.basis = basis
        self.mol = self._build_mol()
        self.results = {}

    def _build_mol(self):
        return gto.Mole(
            atom=self.atoms,
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
            verbose=0,
        ).build()

    def run_rhf(self):
        mf = scf.RHF(mol=self.mol)
        energy = mf.kernel()
        self.results['RHF'] = (energy, mf)
        return energy

    def run_dft(self, functional="b3lyp"):
        mf = scf.RKS(mol=self.mol)
        mf.xc = functional
        energy = mf.kernel()
        self.results['DFT'] = (energy, mf)
        return energy

    def run_fci(self):
        if 'RHF' not in self.results:
            self.run_rhf()
        hf_energy, mf = self.results['RHF']
        cisolver = fci.FCI(mf)
        e_fci, _ = cisolver.kernel()
        self.results['FCI'] = (e_fci, cisolver)
        return e_fci

    def compare_energies(self):
        """
        Return dictionary of correlation energies:
        E_corr = E_method - E_FCI
        """
        if "RHF" not in self.results:
            raise ValueError("Run RHF first...")
        hf_energy, mf = self.results['RHF']
        comparison = {}

        for method in ["DFT", "FCI"]:
            if method in self.results:
                e, _ = self.results[method]
                comparison[method] = e - hf_energy
        return comparison

def separate_identity_term(qubit_operator):
    result = QubitOperator()

    constant = 0
    for term, coefficient in qubit_operator.terms.items():
        if term:
            result += QubitOperator(term, coefficient)
        else:
            constant += coefficient

    return result, constant


# * * * * * * * * * * * * * * * * * * *
#  CALCULATIONS
# * * * * * * * * * * * * * * * * * * *
if __name__ == "__main__":
    # Example for the whole he-series
    bas = "sto-3g"
    atoms = {
        "He" : AtomSystem([("He", [0.0, 0.0, 0.0])], charge=0, spin=0, basis=bas),
        "Li1+": AtomSystem([("Li", [0.0, 0.0, 0.0])], charge=1, spin=0, basis=bas),
        "Be2+":AtomSystem([("Be", [0.0, 0.0, 0.0])], charge=2, spin=0, basis=bas),
        "B3+": AtomSystem([("B", [0.0, 0.0, 0.0])], charge=3, spin=0, basis=bas),
        "C4+": AtomSystem([("C", [0.0, 0.0, 0.0])], charge=4, spin=0, basis=bas),
        "N5+": AtomSystem([("N", [0.0, 0.0, 0.0])], charge=5, spin=0, basis=bas),
        "O6+": AtomSystem([("O", [0.0, 0.0, 0.0])], charge=6, spin=0, basis=bas),
        "F7+": AtomSystem([("F", [0.0, 0.0, 0.0])], charge=7, spin=0, basis=bas),
        "Ne8+":AtomSystem([("Ne", [0.0, 0.0, 0.0])], charge=8, spin=0, basis=bas)
    }

    symbol_lst = ["He", "Li1+", "Be2+", "B3+", "C4+", "N5+", "O6+", "F7+", "Ne8+"]

    fci_corr_lst = []
    dft_corr_lst = []
    e_dft_lst = []
    e_hf_lst = []
    e_fci_lst = []


    for element in symbol_lst:
        print(f"\n========== {element} ==========")

        e_hf = atoms[element].run_rhf()
        e_fci = atoms[element].run_fci()
        e_dft = atoms[element].run_dft("b3lyp")

        print("RHF energy:", e_hf)
        print("FCI energy:", e_fci)
        print("DFT energy:", e_dft)

        corr = atoms[element].compare_energies()
        fci_corr_lst.append(corr["FCI"])
        dft_corr_lst.append(corr["DFT"])

        e_dft_lst.append(e_dft)
        e_hf_lst.append(e_hf)
        e_fci_lst.append(e_fci)

        # Get Integrals
        # Create output directory, and check if it exists
        output_dir = f"./He_series_RHF_integrals_{bas}"

        if not os.path.exists(output_dir):
            print(f"Creating output directory:, {output_dir}")
            os.makedirs(output_dir)
        else:
            print(f"Output directory already exists, {output_dir}")

        mol = atoms[element].mol
        mf = atoms[element].results["RHF"][1]

        # <S^2>
        ss = mf.spin_square()
        print(f"Spin square <S^2>: {ss[0]}")

        print("Orbital idx   Energy")
        for i, (e, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
            print(f"{i + 1:3d}  {e:12.6f}  occ={occ}")


        h0, h1, h2 = get_restricted_ints_full(mol, mf, localized=False)
        filename = f"{element}_{atoms[element].mol.basis}"
        save_restricted_integrals(filename, h0, h1, h2, folder=f"{output_dir}")
