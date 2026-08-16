import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import write

## system
box = np.array([100,100,100])
region = np.array([[0,98],[0,98],[3,100]])                                 # lo, hi for all 3 aixs
sys = Atoms(cell=box,pbc=True)
n_moles = 300

## molecule structure
bond_angle = [0,120,240]
bond_dist = 2.355
angles = np.deg2rad(bond_angle)
positions = [[0, 0, 0],
             [bond_dist*np.cos(angles[0]),bond_dist*np.sin(angles[0]),0],
             [bond_dist*np.cos(angles[1]),bond_dist*np.sin(angles[1]),0],
             [bond_dist*np.cos(angles[2]),bond_dist*np.sin(angles[2]),0]]

crcl3 = Atoms("CrCl3", positions = positions)

## check the molecule
## print(crcl3)
print('molecule poditions are: ', crcl3.positions)
print(crcl3.get_distance(0,1))
print(crcl3.get_distance(0,2))
print(crcl3.get_distance(0,3))
print(crcl3.get_angle(1,0,2))

## engine
rn = np.random.default_rng(42)

## random rotation
def rotation(atoms,rn):
    a = atoms.copy()
    a.rotate(rn.uniform(0,360),"z",center = "COM")       # azimuth angle (rotation)
    theta = np.rad2deg(np.arccos(2 * rn.random() - 1))   
    a.rotate(theta,"x",center="COM")                     # polar angle rotate around the sphere (tilt)
    a.rotate(rn.uniform(0, 360),"z", center="COM")       # rotation around its own axis (spin)
    return a

## random placement

for i in range(n_moles):

    # Randomly rotate molecule
    trial = rotation(crcl3, rn)

    # COM value
    com = trial.get_center_of_mass()

    # Find molecular boundaries
    xmin = np.min(trial.positions[:, 0]) 
    xmax = np.max(trial.positions[:, 0])

    ymin = np.min(trial.positions[:, 1])
    ymax = np.max(trial.positions[:, 1])

    zmin = np.min(trial.positions[:, 2])
    zmax = np.max(trial.positions[:, 2])

    # COM range

    xmin_allowed = region[0,0] + (com[0] - xmin)
    xmax_allowed = region[0,1] - (xmax - com[0])

    ymin_allowed = region[1,0] + (com[1] - ymin)
    ymax_allowed = region[1,1] - (ymax - com[1])

    zmin_allowed = region[2,0] + (com[2] - zmin)
    zmax_allowed = region[2,1] - (zmax - com[2])

    # Random final COM positions

    target_x = rn.uniform(xmin_allowed,xmax_allowed)
    target_y = rn.uniform(ymin_allowed,ymax_allowed)
    target_z = rn.uniform(zmin_allowed,zmax_allowed)

    target = np.array([target_x,target_y,target_z])

    # translation vector

    translation = target - com

    # Move molecule

    trial.translate(translation)

    # molecule to system
    sys += trial

sys.wrap()

# optput files

write("data.lmp", sys, 
      format="lammps-data", 
      specorder=["Cr", "Cl"],    # list of chemical species present 
      units="metal", 
      atom_style="atomic", 
      masses=True,
      )

write("data.xyz", sys)
write("data.cif", sys)