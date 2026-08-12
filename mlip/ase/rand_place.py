import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import write

n_molecules = 50

# ---- template: one H2 molecule ----------------------------------
h2 = molecule("H2")                       # H-H bond ~0.74 A

# ---- empty periodic box -----------------------------------------
box = np.array([30.0, 30.0, 30.0])
system = Atoms(cell=box, pbc=True)

random_number = np.random.default_rng(0)
print(random_number.random(1))

# ---- uniform random orientation ---------------------------------
def random_rotation(atoms,random_number ):
    a = atoms.copy()
    a.rotate(random_number.uniform(0, 360), "z", center="COM")
    theta = np.rad2deg(np.arccos(2 * random_number.random() - 1))
    a.rotate(theta, "x", center="COM")
    a.rotate(random_number.uniform(0, 360), "z", center="COM")
    return a


# ---- place N molecules (no overlap check yet) -------------------
for i in range(n_molecules):
    trial = random_rotation(h2, random_number)
    trial.translate(random_number.random(3) * box - trial.get_center_of_mass())
    system += trial

system.wrap()                                                                               # fold any stray atoms into the box
print(len(system), "atoms |", system.get_chemical_formula())

write("data.h2.lmp", system, format="lammps-data",
      specorder=["H"], units="metal", atom_style="atomic", masses=True)
write("h2_box.xyz", system)