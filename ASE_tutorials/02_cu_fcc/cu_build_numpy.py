import numpy as np
import matplotlib.pyplot as plt

a = 3.61
fcc_uc_posi = np.array([
    [0.0, 0.0, 0.0],
    [0.0, a/2, a/2],
    [a/2, 0.0, a/2],
    [a/2, a/2, 0.0]
])
rep = [5, 5, 5]

uc_vol = a**3
uc_n = len(fcc_uc_posi[:, 0])
cu_MW = 63.55
Na = 6.023e23
rho_uc = (uc_n * cu_MW * 1e24) / (Na * uc_vol)
print("Unit Cell Volume:", uc_vol, "A3")
print("Number of atoms:", uc_n)
print("Unit Cell Density:", rho_uc, "g/cm3")
# Plot unit cell
fig = plt.figure()
ax = fig.add_subplot(1,2,1, projection='3d')
ax.scatter(
    fcc_uc_posi[:, 0],
    fcc_uc_posi[:, 1],
    fcc_uc_posi[:, 2],
    s=50,
    label='Cu',
    color='black'
)
ax.set_title("Cu Unit Cell")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

## supercell
r = []
for i in range(rep[0]):
    for j in range(rep[1]):
        for k in range(rep[2]):
            for atom in fcc_uc_posi:
                posi = atom + np.array([i, j, k])*a
                r.append(posi)
fcc_sc_posi = np.array(r)
sc_vol = a*rep[0] * a*rep[1] * a*rep[2]
sc_n = uc_n * rep[0]*rep[1]*rep[2]
rho_sc = (sc_n * cu_MW * 1e24) / (Na * sc_vol)
print("Super Cell Volume:", sc_vol, "A3")
print("Number of atoms in Sc:", sc_n)
print("Super Cell Density:", rho_sc, "g/cm3")

# Plot supercell
ax1 = fig.add_subplot(1,2,2, projection='3d')

ax1.scatter(
    fcc_sc_posi[:, 0],
    fcc_sc_posi[:, 1],
    fcc_sc_posi[:, 2],
    s=50,
    label='Cu',
    color='black'
)

ax1.set_title("Cu Supercell")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

ax1.legend()

plt.show()