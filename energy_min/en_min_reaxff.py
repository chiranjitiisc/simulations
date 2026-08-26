from ase.io import read, write
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.calculators.lammpslib import LAMMPSlib

atoms = read("unit.cif")

# These are exactly your pair_style / pair_coeff / qeq lines
cmds = [
    "pair_style reaxff NULL safezone 3.0 mincap 150",
    "pair_coeff * * reaxff_mo_al_o_s.lmp Al O",
    "fix qeq all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff",
]

# ReaxFF requires 'real' units and atom_style charge — override the header
header = [
    "units real",
    "atom_style charge",
    "atom_modify map array sort 0 0",
]

atoms.calc = LAMMPSlib(
    lmpcmds=cmds,
    lammps_header=header,
    atom_types={"Al": 1, "O": 2},   # must match the Al O order in pair_coeff
    keep_alive=True,                 # keep the LAMMPS instance + QEq state alive
    log_file="lammps.log",
)

# positions + cell relaxation (the ASE version of fix box/relax tri)
opt = FIRE(FrechetCellFilter(atoms), trajectory="opt.traj")
opt.run(fmax=0.05)   # eV/Å — ASE converts LAMMPS 'real' units to eV/Å for you

write("relaxed.cif", atoms)
print("Energy (eV):", atoms.get_potential_energy())