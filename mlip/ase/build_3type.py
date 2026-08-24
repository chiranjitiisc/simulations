import numpy as np
from ase.io import read, write

# ============================================================
#  Build Mo3O9 / Al2O3 heterostructure -- 3 atom types
#  Al = 1, O = 2, Mo = 3   (molecule oxygen is ordinary O; NO
#  duplicate element, so the CUDA D3 pair style is happy).
#  Track the molecule later by ATOM ID: it is atoms 1..n_mol.
#  Inputs needed in this folder: supercell.cif , mo3o9_single.xyz
# ============================================================

z_len       = 50.0     # box height (vacuum along c)
mol_sub_gap = 2.5      # true gap: highest substrate atom -> lowest molecule atom (A)

# ---- read ----
sub = read("supercell.cif")
mol = read("mo3o9_single.xyz")
n_mol = len(mol)
n_sub = len(sub)

# ---- place molecule above the slab (COM-aligned in x,y; true gap in z) ----
sub_com = sub.get_center_of_mass()
mol_com = mol.get_center_of_mass()
shift_x = sub_com[0] - mol_com[0]
shift_y = sub_com[1] - mol_com[1]
sub_z_max = sub.get_positions()[:, 2].max()
mol_z_min = mol.get_positions()[:, 2].min()
mol.translate([shift_x, shift_y, sub_z_max + mol_sub_gap - mol_z_min])

# ---- combine (molecule first) and set the box (keep substrate a,b; c = vacuum) ----
sys = mol + sub
cell = sub.cell.array.copy()
cell[2] = [0.0, 0.0, z_len]
sys.set_cell(cell)
sys.set_pbc(True)
sys.wrap()

# ---- outputs ----
write("data.xyz", sys)
write("data.cif", sys)
write("data.lmp", sys, format="lammps-data",
      specorder=["Al", "O", "Mo"],       # 3 types, no duplicate element
      units="metal", atom_style="atomic", masses=True)

print(f"molecule = atoms 1..{n_mol} (Mo3O9);  substrate = atoms {n_mol+1}..{n_mol+n_sub}")
