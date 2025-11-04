# pyQCTools

`pyQCTools` is a Python package designed for integration and use within Julia through [PyCall](https://github.com/JuliaPy/PyCall.jl).
It has been specifically designed to work with `QuantumChemQC` package. 
This guide explains how to clone, install, and use the package.

---

## Installation and Instructions

### 1. Clone the repository
```bash
git clone https://github.com/AaronRodriguezJimenez/pyQCTools.git
cd /my_dir/pyQCTools
install -e .
```

### 2. (Recommended) Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate          # on Linux/macOS
```

### 3. Install the package
```bash
pip install -e .
```
Or, alternatively

```bash
/path/to/python -m pip install -e .
```

### 4. Configure Julia and PyCall
```bash
export PYTHON=/path/to/python
julia -e 'using Pkg; Pkg.build("PyCall")'
```

### 5. Using pyQCTools from Julia
```julia
using PyCall
pyqctools = pyimport("pyqctools")
utils = pyimport("pyqctools.utils")
```

Call functions or class from the module:
```julia
# import the module (lowercase)
pq = pyimport("pyqctools")
println("pyqctools imported:", pq)

# run a tiny test with PySCF to validate round-trip mutation
@pyimport pyscf.gto as gto
@pyimport pyscf.scf as scf

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
println("initial basis: ", mol.basis)
pq.set_basis_and_build(mol, "6-31g")
println("new basis: ", mol.basis)

mf = scf.RHF(mol)
energy = pq.compute_scf_energy(mf)
println("SCF energy: ", energy)
```
