import numpy as np
from ase import Atoms
from ase.build import molecule

# ---- step 1: template molecule (the stamp) ----------------------
water = molecule("H2O")          # G2 geometry: O-H 0.9686 A, HOH 104 deg

# ---- step 2: empty periodic box ---------------------------------
box = np.array([50.0, 50.0, 50.0])
system = Atoms(cell=box, pbc=True)

# ---- rng: one generator, created once ---------------------------
rng = np.random.default_rng(42)

# ---- step 4: uniform random orientation -------------------------
def random_rotation(atoms, rng):
    a = atoms.copy()                                   # protect the template
    a.rotate(rng.uniform(0, 360), "z", center="COM")   # spin
    theta = np.rad2deg(np.arccos(2 * rng.random() - 1))
    a.rotate(theta, "x", center="COM")                 # tilt (cos-uniform)
    a.rotate(rng.uniform(0, 360), "z", center="COM")   # spin
    return a

# ---- step 3 + 4 together: orient, then place --------------------
for i in range(5):                       # 5 test stamps
    trial = random_rotation(water, rng)
    trial.translate(rng.random(3) * box - trial.get_center_of_mass())
    system += trial
    print(f"molecule {i+1}: COM = {trial.get_center_of_mass().round(2)}, "
          f"O-H = {trial.get_distance(0, 1):.4f}")

print(len(system), "atoms:", system.get_chemical_formula())