from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.calculators.lj import LennardJones

atoms = read("crystal.cif")

# The entire "force field" is two numbers plus a cutoff
atoms.calc = LennardJones(
    epsilon=0.0103,   # well depth, in eV
    sigma=3.40,       # distance where V(r)=0, in Å
    rc=10.0,          # cutoff radius, Å
    smooth=True,      # taper the potential to 0 at rc (avoids a force jump)
)

opt = BFGS(FrechetCellFilter(atoms), trajectory="opt.traj")
opt.run(fmax=0.01)

write("relaxed.cif", atoms)
print("Energy (eV):", atoms.get_potential_energy())