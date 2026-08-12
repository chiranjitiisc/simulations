import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import write

# ---- template: one H2 molecule ----------------------------------
h2 = molecule("H2")                       # H-H bond ~0.74 A

# ---- empty periodic box -----------------------------------------
box = np.array([30.0, 30.0, 30.0])
system = Atoms(cell=box, pbc=True)

rng = np.random.default_rng(42)

# ---- uniform random orientation ---------------------------------
def random_rotation(atoms, rng):
    a = atoms.copy()
    a.rotate(rng.uniform(0, 360), "z", center="COM")
    theta = np.rad2deg(np.arccos(2 * rng.random() - 1))
    a.rotate(theta, "x", center="COM")
    a.rotate(rng.uniform(0, 360), "z", center="COM")
    return a

# ---- place N molecules (no overlap check yet) -------------------
n_molecules = 50
for i in range(n_molecules):
    trial = random_rotation(h2, rng)
    trial.translate(rng.random(3) * box - trial.get_center_of_mass())
    system += trial

system.wrap()                             # fold any stray atoms into the box
print(len(system), "atoms |", system.get_chemical_formula())
print("H-H of molecule 1:", round(system.get_distance(0, 1), 4), "A")

write("data.lmp", system, format="lammps-data",
      specorder=["H"], units="metal", atom_style="atomic", masses=True)
write("h2_box.xyz", system)
print("wrote data.lmp and h2_box.xyz")
