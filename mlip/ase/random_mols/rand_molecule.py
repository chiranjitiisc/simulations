import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import write

# system
box = np.array([50,50,50])
sys = Atoms(cell=box,pbc=True)
n_moles = 100

# molecule structure
bond_angle = [0,120,240]
bond_dist = 2.355
angles = np.deg2rad(bond_angle)
positions = [[0, 0, 0],
             [bond_dist*np.cos(angles[0]),bond_dist*np.sin(angles[0]),0],
             [bond_dist*np.cos(angles[1]),bond_dist*np.sin(angles[1]),0],
             [bond_dist*np.cos(angles[2]),bond_dist*np.sin(angles[2]),0]]

crcl3 = Atoms("CrCl3", positions = positions)

# check the molecule
print(crcl3)
print(crcl3.positions)
print(crcl3.get_distance(0,1))
print(crcl3.get_distance(0,2))
print(crcl3.get_distance(0,3))
print(crcl3.get_angle(1,0,2))

# engine


## random rotation
rn = np.random.default_rng(42)
def rotation(atoms,rn):
    a = atoms.copy()
    a.rotate(rn.uniform(0,360),"z",center = "COM")       # azimuth angle (rotation)
    theta = np.rad2deg(np.arccos(2 * rn.random() - 1))   
    a.rotate(theta,"x",center="COM")                     # polar angle rotate around the sphere (tilt)
    a.rotate(rn.uniform(0, 360),"z", center="COM")       # rotation around its own axis (spin)
    return a
## random placement
for i in range(n_moles):
    trial = rotation(crcl3,rn)
    trial.translate(rn.random(3)*box - trial.get_center_of_mass())
    sys += trial
sys.wrap()



# write optput files

write("data.lmp", sys, 
      format="lammps-data", 
      specorder=["Cr", "Cl"],    # list of chemical species present 
      units="metal", 
      atom_style="atomic", 
      masses=True)

write("h2_box.xyz", sys)