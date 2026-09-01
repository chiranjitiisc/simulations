import numpy as np
import math
import matplotlib.pyplot as plt

posi = np.array([[0.0000, -0.060370, 0.0000],      # O
                 [0.68558, 0.502500, 0.0000],      # H
                 [-0.68558, 0.502500, 0.0000]])    # H

def bond_length(posi):
    dist_oh = (posi[0] - posi[1])
    bl = np.linalg.norm(dist_oh)
    #bl = ((dist_oh[0])**2 + (dist_oh[1])**2 + (dist_oh[2])**2 )**0.5
    return(bl)
def bond_angle(posi):
    oh1 = (posi[1] - posi[0])
    oh2 = (posi[2] - posi[0])
    bl1 = np.linalg.norm(oh1)
    bl2 = np.linalg.norm(oh2)
    # bl1 = ((oh1[0])**2 + (oh1[1])**2 + (oh1[2])**2 )**0.5 
    # bl2 = ((oh2[0])**2 + (oh2[1])**2 + (oh2[2])**2 )**0.5 
    oh1_dot_oh2  = np.dot(oh1,oh2)
    theta = math.acos(oh1_dot_oh2/(bl1*bl2))
    return(math.degrees(theta))

print("Bond length is: ", bond_length(posi), 'A')
print("Bond Angle is: ", bond_angle(posi), "degrees")

## plot
fig = plt.figure()
ax = fig.add_subplot(projection ='3d')

# Plot oxygen
ax.scatter(
    posi[0, 0],
    posi[0, 1],
    posi[0, 2],
    s=500,
    label="O",
    color = "red"
)
# Plot hydrogens
ax.scatter(
    posi[1:, 0],
    posi[1:, 1],
    posi[1:, 2],
    s=100,
    label="H",
    color ="green"
)

# O-H bonds
ax.plot(
    [posi[0, 0], posi[1, 0]],
    [posi[0, 1], posi[1, 1]],
    [posi[0, 2], posi[1, 2]]
)

ax.plot(
    [posi[0, 0], posi[2, 0]],
    [posi[0, 1], posi[2, 1]],
    [posi[0, 2], posi[2, 2]]
)

# Labels
ax.text(posi[0, 0], posi[0, 1], posi[0, 2], "O")
ax.text(posi[1, 0], posi[1, 1], posi[1, 2], "H")
ax.text(posi[2, 0], posi[2, 1], posi[2, 2], "H")

ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")

ax.set_title("H₂O Molecule")
ax.legend()
plt.show()
