"""
Energy + cell minimization of the F-mica unit cell with a *local* MACE model
(all_elements_grid_25_ep120_lr5e-4_bs01_fw100_compiled.model)

The .model file is a TorchScript-serialised ScaleShiftMACE (the kind produced by
MACE's create_lammps_model.py), stored in float64. So we point MACECalculator at
the file directly instead of downloading a foundation model with mace_mp().
"""

import os
import torch

from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixSymmetry
from mace.calculators import MACECalculator

# ---------------------------------------------------------------- settings ---
MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "all_elements_grid_25_ep120_lr5e-4_bs01_fw100_compiled.model",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "float64"          # this compiled model stores double-precision weights
FMAX = 0.02                # eV/Ang
# ------------------------------------------------------------------------------


def build_calculator():
    """MACECalculator, with a fallback for TorchScript archives.

    Newer torch versions default to weights_only=True in torch.load, which
    refuses a TorchScript archive. If that happens we load it explicitly with
    torch.jit.load and hand the ready model to MACECalculator.
    """
    try:
        return MACECalculator(
            model_paths=MODEL,
            device=DEVICE,
            default_dtype=DTYPE,
        )
    except Exception as err:
        print(f"[info] direct load failed ({type(err).__name__}: {err})")
        print("[info] retrying with torch.jit.load ...")
        model = torch.jit.load(MODEL, map_location=DEVICE)
        model = model.double() if DTYPE == "float64" else model.float()
        for p in model.parameters():
            p.requires_grad = False
        return MACECalculator(models=[model], device=DEVICE, default_dtype=DTYPE)


atoms = read("fmica_unit_cell.cif")
atoms.calc = build_calculator()

# (optional) keep the space-group symmetry during relaxation
atoms.set_constraint(FixSymmetry(atoms))

# relax positions AND cell together
opt = BFGS(FrechetCellFilter(atoms), trajectory="optimization.traj")
opt.run(fmax=FMAX)

write("optimized.cif", atoms)
print("Final energy:", atoms.get_potential_energy(), "eV")
print("Final cell:", atoms.cell.cellpar())
