import numpy as np
import math
import matplotlib.pyplot as plt
from ase.visualize import view

from ase import Atoms
from ase.io import read,write

posi = posi = np.array([[0.0000, -0.060370, 0.0000],      # O
                        [0.68558, 0.502500, 0.0000],      # H
                        [-0.68558, 0.502500, 0.0000]])    # H

h2o = Atoms(symbols = ['O','H','H'], positions = posi)
print("Cmhemical Formula is: ",h2o.get_chemical_formula())
print("Angle is: ", h2o.get_angle(1,0,2), "degrees")
print("Bond length is: ", h2o.get_distance(0,1), "A")

view(h2o)

