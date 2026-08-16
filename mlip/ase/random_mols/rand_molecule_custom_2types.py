import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import write

## system
box = np.array([100,100,100])
region1 = np.array([[40,65],[0,98],[3,100]])                                 # lo, hi for all 3 aixs for mol1
region2 = np.array([[15,35],[0,98],[3,100]])                                 # for mol2
sys = Atoms(cell=box,pbc=True)
n_moles1 = 300
n_moles2 = 150

## molecule1 structure
bond_angle = [0,120,240]
bond_dist = 2.355
angles = np.deg2rad(bond_angle)
positions1 = [[0, 0, 0],
             [bond_dist*np.cos(angles[0]),bond_dist*np.sin(angles[0]),0],
             [bond_dist*np.cos(angles[1]),bond_dist*np.sin(angles[1]),0],
             [bond_dist*np.cos(angles[2]),bond_dist*np.sin(angles[2]),0]]
crcl3 = Atoms("CrCl3", positions = positions1)

## molecule 2 structure
positions2 = [[0.00, 0.00, 0.00],[0.00,0.755,-0.589],[0.00,-0.755,-0.589]]
h2o = Atoms(['O','H','H'],positions = positions2 )

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

for i in range(n_moles1 + n_moles2):

    # Randomly rotate molecule
    trial1 = rotation(crcl3, rn)
    trial2 = rotation(h2o,rn)

    # COM value
    
    com1 = trial1.get_center_of_mass()
    com2 = trial2.get_center_of_mass()

    # Find molecular boundaries
    
    # for mol1
    xmin1 = np.min(trial1.positions[:, 0]) 
    xmax1 = np.max(trial1.positions[:, 0])

    ymin1 = np.min(trial1.positions[:, 1])
    ymax1 = np.max(trial1.positions[:, 1])

    zmin1 = np.min(trial1.positions[:, 2])
    zmax1 = np.max(trial1.positions[:, 2])

    # for 2nd mol
    xmin2 = np.min(trial2.positions[:, 0]) 
    xmax2 = np.max(trial2.positions[:, 0])
    
    ymin2 = np.min(trial2.positions[:, 1])
    ymax2 = np.max(trial2.positions[:, 1])
    
    zmin2 = np.min(trial2.positions[:, 2])
    zmax2 = np.max(trial2.positions[:, 2])


    # COM range
    # MOL 1
    xmin1_allowed = region1[0,0] + (com1[0] - xmin1)
    xmax1_allowed = region1[0,1] - (xmax1 - com1[0])

    ymin1_allowed = region1[1,0] + (com1[1] - ymin1)
    ymax1_allowed = region1[1,1] - (ymax1 - com1[1])

    zmin1_allowed = region1[2,0] + (com1[2] - zmin1)
    zmax1_allowed = region1[2,1] - (zmax1 - com1[2])
    # MOL 2
    xmin2_allowed = region2[0,0] + (com2[0] - xmin2)
    xmax2_allowed = region2[0,1] - (xmax2 - com2[0])
    
    ymin2_allowed = region2[1,0] + (com2[1] - ymin2)
    ymax2_allowed = region2[1,1] - (ymax2 - com2[1])
    
    zmin2_allowed = region2[2,0] + (com2[2] - zmin2)
    zmax2_allowed = region2[2,1] - (zmax2 - com2[2])
        

    # Random final COM positions

    target1_x = rn.uniform(xmin1_allowed,xmax1_allowed)
    target1_y = rn.uniform(ymin1_allowed,ymax1_allowed)
    target1_z = rn.uniform(zmin1_allowed,zmax1_allowed)

    target2_x = rn.uniform(xmin2_allowed,xmax2_allowed)
    target2_y = rn.uniform(ymin2_allowed,ymax2_allowed)
    target2_z = rn.uniform(zmin2_allowed,zmax2_allowed)

    target1 = np.array([target1_x,target1_y,target1_z])
    target2 = np.array([target2_x,target2_y,target2_z])

    # translation vector

    translation1 = target1 - com1
    translation2 = target2 - com2

    # Move molecule

    trial1.translate(translation1)
    trial2.translate(translation2)

    # molecule to system
    sys += trial1
    sys += trial2

sys.wrap()

# optput files

write("data.lmp", sys, 
      format="lammps-data", 
      specorder=["Cr", "Cl","O","H","H"],    # list of chemical species present 
      units="metal", 
      atom_style="atomic", 
      masses=True,
      )

write("data.xyz", sys)
write("data.cif", sys)