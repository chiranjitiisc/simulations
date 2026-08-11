"""
Part 2 capstone — teach a neural network a potential energy surface,
then recover the FORCE from it with autograd.

This is the whole MLIP idea in one dimension:
    * the network is the ENERGY  E(r)
    * the FORCE is -dE/dr, obtained by autograd (never a second network)
    * training on forces (force-matching) needs create_graph=True

We use a Morse potential as the "ground truth" (it stands in for DFT), because
we know its energy AND its analytic force, so we can check everything.

Run:  python3 part2_foundations.py
It prints a results table and saves 'part2_results.png'.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")           # headless backend (no display needed)
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)   # float64: energies/forces want the precision


# ----------------------------------------------------------------------
# 0. The ground-truth potential (our stand-in for DFT)
#    Morse:  E(r) = De*(1 - e)^2 - De,   e = exp(-a (r - re))
#    Force:  F(r) = -dE/dr = 2*De*a*e*(e - 1)
# ----------------------------------------------------------------------
De, a, re = 4.0, 1.7, 1.2       # well depth (eV), width (1/A), equilibrium bond length (A)

def morse_energy(r):
    e = np.exp(-a * (r - re))
    return De * (1.0 - e) ** 2 - De

def morse_force(r):
    e = np.exp(-a * (r - re))
    return 2.0 * De * a * e * (e - 1.0)     # = -dE/dr


# ----------------------------------------------------------------------
# 1. Data.  DENSE grid = the truth we test against (generalization).
#    SPARSE set = the few labels we actually train on. Deliberately few,
#    so that energy-only training leaves the force (the derivative) wobbly
#    between points — which is exactly what force-matching then fixes.
# ----------------------------------------------------------------------
r_lo, r_hi = 0.75, 3.0

r_dense = np.linspace(r_lo, r_hi, 400)
E_dense = morse_energy(r_dense)
F_dense = morse_force(r_dense)

n_train = 18
r_train_np = np.linspace(r_lo, r_hi, n_train)
E_train_np = morse_energy(r_train_np)
F_train_np = morse_force(r_train_np)

# to tensors, shape (M, 1)
r_train = torch.tensor(r_train_np).reshape(-1, 1)
E_train = torch.tensor(E_train_np).reshape(-1, 1)
F_train = torch.tensor(F_train_np).reshape(-1, 1)


# ----------------------------------------------------------------------
# 2. The model: a tiny MLP that maps r -> E.  SiLU (smooth) activations,
#    NOT ReLU: a kink in E would mean a jump in the force. Smoothness is a
#    physics requirement here, not an aesthetic choice.
# ----------------------------------------------------------------------
class EnergyMLP(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 1),
        )
    def forward(self, r):
        return self.net(r)          # shape (M, 1)


def forces_from(model, r):
    """F = -dE/dr for every sample at once.
    Trick: d(sum_k E_k)/dr_j = dE_j/dr_j because sample j's energy
    depends only on sample j's input. create_graph=True keeps this
    differentiable so we can BACKPROP the force loss into the weights."""
    r = r.clone().requires_grad_(True)
    E = model(r)
    dEdr = torch.autograd.grad(E.sum(), r, create_graph=True)[0]
    return -dEdr


# ----------------------------------------------------------------------
# 3. The training loop — the same five lines you will use forever.
#    If w_force > 0 we add the force-matching term (this is the ONLY
#    difference between a plain regressor and an MLIP trainer).
# ----------------------------------------------------------------------
def train(model, epochs=4000, lr=1e-3, w_energy=1.0, w_force=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        opt.zero_grad()                       # 1. clear old gradients (.grad accumulates!)
        E_pred = model(r_train)               # 2. forward: predict energies
        loss = w_energy * ((E_pred - E_train) ** 2).mean()
        if w_force > 0.0:                      #    optional force-matching term
            F_pred = forces_from(model, r_train)
            loss = loss + w_force * ((F_pred - F_train) ** 2).mean()
        loss.backward()                        # 3. d(loss)/d(weights)  (2nd-order if forces are in it)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # tame occasional Adam spikes
        opt.step()                             # 4. nudge the weights
        history.append(loss.item())            # 5. (just bookkeeping)
    return history


def evaluate(model):
    """Energy MSE and force MAE on the DENSE grid (data the model never saw)."""
    r = torch.tensor(r_dense).reshape(-1, 1)
    with torch.no_grad():
        E_pred = model(r).squeeze(1).numpy()
    F_pred = forces_from(model, r).detach().squeeze(1).numpy()
    e_mse = float(np.mean((E_pred - E_dense) ** 2))
    f_mae = float(np.mean(np.abs(F_pred - F_dense)))
    return E_pred, F_pred, e_mse, f_mae


# ----------------------------------------------------------------------
# 4. Run both experiments from the SAME initial weights (fair comparison).
# ----------------------------------------------------------------------
torch.manual_seed(0)
model_E = EnergyMLP()
hist_E = train(model_E, w_force=0.0)                 # energy only
E_pred_E, F_pred_E, emse_E, fmae_E = evaluate(model_E)

torch.manual_seed(0)                                 # identical init
model_F = EnergyMLP()
hist_F = train(model_F, w_force=1.0)                 # energy + force matching
E_pred_F, F_pred_F, emse_F, fmae_F = evaluate(model_F)

print("=" * 60)
print("Generalization on 400 unseen points (trained on only %d):" % n_train)
print("-" * 60)
print("%-22s %14s %14s" % ("model", "energy MSE", "force MAE"))
print("%-22s %14.5f %14.5f" % ("energy-only", emse_E, fmae_E))
print("%-22s %14.5f %14.5f" % ("energy + force", emse_F, fmae_F))
print("-" * 60)
print("force MAE improved %.1fx by adding force-matching" % (fmae_E / fmae_F))
print("=" * 60)


# ----------------------------------------------------------------------
# 5. Figure (colorblind-safe palette: truth=black, energy-only=blue,
#    force-matched=orange).
# ----------------------------------------------------------------------
INK, BLUE, ORANGE = "#0b0b0b", "#2a78d6", "#eb6834"
GRID, MUTED, SURF = "#e1e0d9", "#898781", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# Panel A: energy — both learned curves overlap the truth (energy is the "easy" part)
ax[0,0].plot(r_dense, E_dense, color=INK, lw=2.6, label="true Morse E(r)")
ax[0,0].plot(r_dense, E_pred_E, color=BLUE, lw=2, label="learned (energy-only)")
ax[0,0].plot(r_dense, E_pred_F, color=ORANGE, lw=2, label="learned (energy+force)")
ax[0,0].scatter(r_train_np, E_train_np, color=MUTED, s=26, zorder=5,
                label="the %d training points" % n_train)
ax[0,0].set_title("Energy  E(r)  — both models fit energy well")
ax[0,0].set_xlabel("r  (A)"); ax[0,0].set_ylabel("energy (eV)")
ax[0,0].legend(frameon=False, fontsize=9)

# Panel B: force over the full range — autograd recovers the whole force curve
ax[0,1].plot(r_dense, F_dense, color=INK, lw=2.6, label="true force -dE/dr")
ax[0,1].plot(r_dense, F_pred_E, color=BLUE, lw=2, label="energy-only")
ax[0,1].plot(r_dense, F_pred_F, color=ORANGE, lw=2, label="energy+force")
ax[0,1].set_title("Force  F(r) = -dE/dr   (full range, from autograd)")
ax[0,1].set_xlabel("r  (A)"); ax[0,1].set_ylabel("force (eV/A)")
ax[0,1].legend(frameon=False, fontsize=9)

# Panel C: force zoomed to the chemically relevant region — the wobble is now visible
m = r_dense >= 1.05
ax[1,0].plot(r_dense[m], F_dense[m], color=INK, lw=2.6, label="true force")
ax[1,0].plot(r_dense[m], F_pred_E[m], color=BLUE, lw=2, label="energy-only (wobbles)")
ax[1,0].plot(r_dense[m], F_pred_F[m], color=ORANGE, lw=2, label="energy+force (clean)")
ax[1,0].set_title("Same force, zoomed to r > 1.05 A  — energy-only wanders")
ax[1,0].set_xlabel("r  (A)"); ax[1,0].set_ylabel("force (eV/A)")
ax[1,0].legend(frameon=False, fontsize=9)

# Panel D: force error across r — quantifies why force-matching wins
err_E = np.abs(F_pred_E - F_dense) + 1e-9
err_F = np.abs(F_pred_F - F_dense) + 1e-9
ax[1,1].plot(r_dense, err_E, color=BLUE, lw=2, label="energy-only  |dF|")
ax[1,1].plot(r_dense, err_F, color=ORANGE, lw=2, label="energy+force  |dF|")
ax[1,1].set_yscale("log")
ax[1,1].set_title("Force error  |F_pred - F_true|  (energy-only is worse ~everywhere)")
ax[1,1].set_xlabel("r  (A)"); ax[1,1].set_ylabel("abs force error (eV/A, log)")
ax[1,1].legend(frameon=False, fontsize=9)

fig.suptitle("Part 2 capstone: a network learns E(r); autograd gives the force  "
             "(force-matching cuts force error %.1fx)" % (fmae_E / fmae_F),
             fontsize=13, y=1.005)
fig.tight_layout()
fig.savefig("part2_results.png", dpi=140, bbox_inches="tight")
print("saved part2_results.png")