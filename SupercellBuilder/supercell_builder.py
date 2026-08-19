
# ======================================================================
#                           USER SETTINGS
# ======================================================================

# --- Input -----------------------------------------------------------
CIF_NAME = "mp-48_C.cif"   # just the file NAME (same folder as this script)

# --- Replication -----------------------------------------------------
# Method "diagonal": simple integer tiling along a, b, c (most common).
#   NX, NY, NZ  ->  supercell = unit * (NX, NY, NZ)
# Method "matrix": general transformation, new_cell = P @ old_cell.
#   Use for rotated / orthogonalized / sheared cells. Rows must be integers.
METHOD = "diagonal"          # "diagonal" or "matrix"

NX, NY, NZ = 8, 8, 1         # used when METHOD == "diagonal"

P_MATRIX = [                 # used when METHOD == "matrix"
    [2, 0, 0],
    [1, 2, 0],
    [0, 0, 1],
]

# --- Output ----------------------------------------------------------
OUTPUT_NAME = "supercell"            # base name for output files (written here)
# Structure formats to write. ASE auto-detects from extension.
# Common: "cif", "xyz", "vasp" (POSCAR), "lammps-data".
OUTPUT_FORMATS = ["cif", "xyz", "vasp"]
WRAP_ATOMS = True                    # wrap atoms back into the cell

# ======================================================================
#                     (no need to edit below here)
# ======================================================================

import os
import numpy as np
from ase.io import read, write
from ase.build import make_supercell

# Resolve everything relative to THIS script's folder, so you only need the
# CIF file NAME (not a full path) as long as it sits next to this script.
HERE = os.path.dirname(os.path.abspath(__file__))


def cellpar_block(atoms, title):
    """Return a formatted text block of lattice parameters for an Atoms object."""
    a, b, c, al, be, ga = atoms.cell.cellpar()
    vol = atoms.get_volume()
    n = len(atoms)
    masses = atoms.get_masses().sum()              # amu
    density = masses * 1.66053906660 / vol         # amu/A^3 -> g/cm^3
    sym = atoms.get_chemical_symbols()
    counts = {s: sym.count(s) for s in sorted(set(sym))}
    comp = ", ".join(f"{s}: {c}" for s, c in counts.items())

    lines = [
        f"  {title}",
        f"    chemical formula : {atoms.get_chemical_formula()}",
        f"    composition      : {comp}",
        f"    number of atoms  : {n}",
        f"    a = {a:.6f} A     b = {b:.6f} A     c = {c:.6f} A",
        f"    alpha = {al:.4f} deg   beta = {be:.4f} deg   gamma = {ga:.4f} deg",
        f"    cell volume      : {vol:.6f} A^3",
        f"    mass density     : {density:.4f} g/cm^3",
        "    lattice vectors (A):",
    ]
    for i, v in enumerate(atoms.cell.array):
        lines.append(f"      a{i+1} = [{v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}]")
    return "\n".join(lines)


def main():
    cif_path = os.path.join(HERE, CIF_NAME)
    if not os.path.isfile(cif_path):
        raise FileNotFoundError(
            f"Could not find '{CIF_NAME}' next to this script (looked in {HERE}). "
            "Put the CIF and this script in the same folder, and set CIF_NAME "
            "to just the file name."
        )

    # 1. Read the unit cell ------------------------------------------------
    unit = read(cif_path)
    unit.pbc = True

    # 2. Build the supercell ----------------------------------------------
    if METHOD == "diagonal":
        rep = (NX, NY, NZ)
        supercell = unit * rep
        P = np.diag(rep)
        rep_desc = f"diagonal repeat {NX} x {NY} x {NZ}"
    elif METHOD == "matrix":
        P = np.array(P_MATRIX, dtype=int)
        supercell = make_supercell(unit, P)
        rep_desc = f"transformation matrix P (det = {int(round(np.linalg.det(P)))})"
    else:
        raise ValueError("METHOD must be 'diagonal' or 'matrix'")

    if WRAP_ATOMS:
        supercell.wrap()

    n_cells = int(round(abs(np.linalg.det(P))))

    # 3. Write structure files (beside the script) ------------------------
    base = os.path.join(HERE, OUTPUT_NAME)
    ext = {"vasp": "vasp", "lammps-data": "lmp"}
    written = []
    for fmt in OUTPUT_FORMATS:
        suffix = ext.get(fmt, fmt)
        path = f"{base}.{suffix}"
        write(path, supercell, format=fmt)
        written.append(os.path.basename(path))

    # 4. Write the .txt report --------------------------------------------
    report = os.path.join(HERE, f"{OUTPUT_NAME}_report.txt")
    with open(report, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SUPERCELL BUILD REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"input CIF              : {CIF_NAME}\n")
        f.write(f"replication method     : {rep_desc}\n")
        f.write(f"unit cells in supercell: {n_cells}\n")
        f.write("transformation matrix P (new_cell = P @ old_cell):\n")
        for row in P:
            f.write(f"    [{int(row[0]):3d} {int(row[1]):3d} {int(row[2]):3d}]\n")
        f.write(f"atoms: {len(unit)} (unit cell)  ->  {len(supercell)} (supercell)\n\n")

        f.write("-" * 70 + "\n")
        f.write(cellpar_block(unit, "UNIT CELL") + "\n\n")
        f.write("-" * 70 + "\n")
        f.write(cellpar_block(supercell, "SUPERCELL") + "\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("output structure files:\n")
        for p in written:
            f.write(f"    {p}\n")

    # 5. Echo a short summary to the screen -------------------------------
    print(f"Loaded {CIF_NAME}: {unit.get_chemical_formula()} ({len(unit)} atoms)")
    print(f"Built supercell: {supercell.get_chemical_formula()} "
          f"({len(supercell)} atoms, {n_cells} unit cells)")
    print("Wrote:")
    for p in written + [os.path.basename(report)]:
        print(f"    {p}")


if __name__ == "__main__":
    main()
