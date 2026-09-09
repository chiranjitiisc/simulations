import numpy as np
from ase.io import read, write
from ase import Atoms

gbl = read("gbl.pdb")
pbi2 = read("pbi2.xyz")
mai = read("mai.xyz")

box = [48,48,48]

n_gbl = 112
n_pbi2 = 200
n_mai = 200

region1 = np.array([[0,box[0]],[0, box[1]], [0,box[2]]])
region2 = np.array([[0,box[0]],[0, box[1]], [0,box[2]]])
region3 = np.array([[0,box[0]],[0, box[1]], [0,box[2]]])

sys = Atoms(cell = box, pbc = True)

rn = np.random.default_rng(42)

def rotation(atoms,rn):
    a = atoms.copy()
    a.rotate(rn.uniform(0,360),"z",center = "COM")       # azimuth angle (rotation)
    theta = np.rad2deg(np.arccos(2 * rn.random() - 1))   
    a.rotate(theta,"x",center="COM")                     # polar angle rotate around the sphere (tilt)
    a.rotate(rn.uniform(0, 360),"z", center="COM")       # rotation around its own axis (spin)
    return a

for i in range(n_gbl + n_pbi2 + n_mai):

    trial1 = rotation(gbl, rn)
    trial2 = rotation(pbi2,rn)
    trial3 = rotation(mai,rn)
    
    # COM value
    com1 = trial1.get_center_of_mass()
    com2 = trial2.get_center_of_mass()
    com3 = trial3.get_center_of_mass() 
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

    # for 3nd mol
    xmin3 = np.min(trial3.positions[:, 0]) 
    xmax3 = np.max(trial3.positions[:, 0])
    
    ymin3 = np.min(trial3.positions[:, 1])
    ymax3 = np.max(trial3.positions[:, 1])
    
    zmin3 = np.min(trial3.positions[:, 2])
    zmax3 = np.max(trial3.positions[:, 2])


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

    xmin3_allowed = region3[0,0] + (com3[0] - xmin3)
    xmax3_allowed = region3[0,1] - (xmax3 - com3[0])
    
    ymin3_allowed = region3[1,0] + (com3[1] - ymin3)
    ymax3_allowed = region3[1,1] - (ymax3 - com3[1])
    
    zmin3_allowed = region3[2,0] + (com3[2] - zmin3)
    zmax3_allowed = region3[2,1] - (zmax3 - com3[2])
        

    # Random final COM positions

    target1_x = rn.uniform(xmin1_allowed,xmax1_allowed)
    target1_y = rn.uniform(ymin1_allowed,ymax1_allowed)
    target1_z = rn.uniform(zmin1_allowed,zmax1_allowed)

    target2_x = rn.uniform(xmin2_allowed,xmax2_allowed)
    target2_y = rn.uniform(ymin2_allowed,ymax2_allowed)
    target2_z = rn.uniform(zmin2_allowed,zmax2_allowed)
    
    target3_x = rn.uniform(xmin3_allowed,xmax3_allowed)
    target3_y = rn.uniform(ymin3_allowed,ymax3_allowed)
    target3_z = rn.uniform(zmin3_allowed,zmax3_allowed)

    target1 = np.array([target1_x,target1_y,target1_z])
    target2 = np.array([target2_x,target2_y,target2_z])
    target3 = np.array([target3_x,target3_y,target3_z])
    # translation vector

    translation1 = target1 - com1
    translation2 = target2 - com2
    translation3 = target3 - com3

    # Move molecule

    trial1.translate(translation1)
    trial2.translate(translation2)
    trial3.translate(translation3)

    # molecule to system
    sys += trial1
    sys += trial2
    sys += trial3

sys.wrap()
print(sys.get_chemical_formula())
write("data.lmp", sys, format="lammps-data",specorder=["C", "H", "I","N","O","Pb"], units="real", atom_style="full", masses=True)

