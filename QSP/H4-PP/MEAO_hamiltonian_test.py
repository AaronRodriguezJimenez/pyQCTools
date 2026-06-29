from MEAO.meao_Hamiltonian import build_meao_hamiltonian


ham = build_meao_hamiltonian(
        atom="N 0 0 0; N 0 0 1.098",
        basis="ccpvdz",
        spin=0,
    )

print("Number of MEAOs:", ham["norb"])
print("Number of electrons:", ham["nelec"])
print("MEAO bonds:", ham["bonds"])
print("h1e shape:", ham["h1e"].shape)
print("eri shape:", ham["eri"].shape)