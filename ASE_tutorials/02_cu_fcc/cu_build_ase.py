import numpy as np
from ase import Atoms
from ase.visualize import view
from ase.build import bulk, make_supercell
from ase.io import read

a = 3.61
# cu_uc = bulk('Cu','fcc',a) gives primitive unitcell (1 lattice points)

cu_uc = read("Cu_mp-30_conventional_standard.cif")

# fcc_uc_posi = np.array([
#     [0.0, 0.0, 0.0],
#     [0.0, a/2, a/2],
#     [a/2, 0.0, a/2],
#     [a/2, a/2, 0.0]
# ])
# cu_uc = Atoms(symbols = ['Cu','Cu','Cu','Cu'],
#                positions = fcc_uc_posi,
#                cell = [a,a,a],
#                pbc = True)


print("Number of Atoms is Super cell: ", len(cu_uc))
print("Volume of Super cell: ", cu_uc.get_volume())

cu_sc = cu_uc.repeat([5,5,5])
print("The Atomic formula is: ", cu_uc.get_chemical_symbols())
print("Lattice vectors of the unit cell: ", cu_uc.cell.cellpar())  ### for a,b,c and alpha beta and gamma
print("Number of Atoms is Super cell: ", len(cu_sc))
print("Volume of Super cell: ", cu_sc.get_volume())


view(cu_sc)

