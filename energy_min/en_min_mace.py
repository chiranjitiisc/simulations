from ase.optimize import BFGS
from ase.io import read, write
from ase.filters import FrechetCellFilter
from ase.constraints import FixSymmetry
from mace.calculators import mace_mp

atoms = read("crystal.cif")
atoms.calc = mace_mp(model="small-omat-0", device="cpu")

# (optional) keep the space-group symmetry during relaxation
atoms.set_constraint(FixSymmetry(atoms))

# relax positions AND cell together
opt = BFGS(FrechetCellFilter(atoms), trajectory="optimization.traj")
opt.run(fmax=0.02)

write("optimized.cif", atoms)
print("Final energy:", atoms.get_potential_energy(), "eV")