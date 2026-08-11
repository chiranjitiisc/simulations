"""
Autograd deep-dive problem set — WORKED SOLUTIONS (self-checking).

Focus: what `r` is (leaf tensors, requires_grad) and how `E.backward()` works
(the graph, the chain rule, .grad). Ten problems.

Try them yourself first (see Part2_Autograd_Problems.md), then run:
    python3 part2_autograd_solutions.py
Every problem asserts its own correctness and prints [PASS]. A clean run means
all ten are right.
"""

import math
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

def check(name, ok, extra=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    assert ok, f"{name} FAILED"


# =====================================================================
# PROBLEM 1 — Anatomy of a tensor: leaf, requires_grad, grad, and propagation
# Predict each attribute BEFORE running.
# =====================================================================
def problem_1():
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    check("P1 x is a leaf",            x.is_leaf is True)
    check("P1 x requires grad",        x.requires_grad is True)
    check("P1 x.grad empty at first",  x.grad is None)

    y = x * 2                       # y is a RESULT of an op...
    check("P1 y is NOT a leaf",       y.is_leaf is False)
    check("P1 y requires grad (inherited)", y.requires_grad is True)
    check("P1 y has a grad_fn",       y.grad_fn is not None)

    z = torch.tensor([5.0])         # no requires_grad
    w = z * 2
    check("P1 no-grad input -> output doesn't track",
          (w.requires_grad is False) and (w.grad_fn is None))


# =====================================================================
# PROBLEM 2 — The graph mirrors the forward ops in reverse
# For E = 4*(r**-12 - r**-6): last op is Mul -> Sub -> two Pow.
# =====================================================================
def problem_2():
    r = torch.tensor([1.2], requires_grad=True)
    E = 4.0 * (r**-12 - r**-6)

    check("P2 last op is a multiply", "Mul" in type(E.grad_fn).__name__)
    subs = [nf[0] for nf in E.grad_fn.next_functions if nf[0] is not None]
    check("P2 its parent is a subtract", any("Sub" in type(s).__name__ for s in subs))
    sub = [s for s in subs if "Sub" in type(s).__name__][0]
    pows = [nf[0] for nf in sub.next_functions if nf[0] is not None]
    check("P2 subtract has two power parents",
          len(pows) == 2 and all("Pow" in type(p).__name__ for p in pows))


# =====================================================================
# PROBLEM 3 — backward IS the chain rule. Do it by hand, then check.
# f = (3x+1)^2 at x=2  ->  df/dx = 2(3x+1)*3 = 6*(3*2+1) = 42
# =====================================================================
def problem_3():
    x = torch.tensor(2.0, requires_grad=True)
    f = (3 * x + 1) ** 2
    f.backward()
    check("P3 backward == hand chain rule", abs(x.grad.item() - 42.0) < 1e-12,
          f"x.grad = {x.grad.item()}")


# =====================================================================
# PROBLEM 4 — Contributions from multiple paths SUM
# (This is exactly why r fed BOTH r**-12 and r**-6 in the LJ example.)
# =====================================================================
def problem_4():
    # (a) x appears in two separate terms: dy/dx = 2x + e^x
    x = torch.tensor(1.0, requires_grad=True)
    y = x**2 + torch.exp(x)
    y.backward()
    check("P4a two-term sum", abs(x.grad.item() - (2 * 1.0 + math.e)) < 1e-12,
          f"x.grad = {x.grad.item():.6f}")

    # (b) x appears in two FACTORS: y=(x^2)(x+1) -> product rule = paths summed
    #     dy/dx = 2x(x+1) + x^2 ; at x=2 -> 12 + 4 = 16
    x = torch.tensor(2.0, requires_grad=True)
    y = (x**2) * (x + 1)
    y.backward()
    check("P4b two-factor product (paths sum)", abs(x.grad.item() - 16.0) < 1e-12,
          f"x.grad = {x.grad.item()}")


# =====================================================================
# PROBLEM 5 — Reproduce the LJ force at a NEW distance
# E = 4(r^-12 - r^-6); dE/dr = -48 r^-13 + 24 r^-7; force = -dE/dr
# =====================================================================
def problem_5():
    r = torch.tensor([1.15], requires_grad=True)
    E = 4.0 * (r**-12 - r**-6)
    E.backward()
    rn = 1.15
    analytic = -48 * rn**-13 + 24 * rn**-7
    check("P5 dE/dr matches analytic", abs(r.grad.item() - analytic) < 1e-10,
          f"dE/dr = {r.grad.item():.5f}")
    force = -r.grad.item()
    check("P5 force = -dE/dr", abs(force + analytic) < 1e-10, f"force = {force:.5f} eV/A")


# =====================================================================
# PROBLEM 6 — .grad ACCUMULATES across backward() calls
# =====================================================================
def problem_6():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    (x**2).sum().backward()                 # grad = 2x = [2, 4]
    first = x.grad.clone()
    check("P6 first backward -> 2x", torch.allclose(first, torch.tensor([2.0, 4.0])))

    (x**2).sum().backward()                 # NOT zeroed -> doubles to [4, 8]
    check("P6 second backward accumulates", torch.allclose(x.grad, 2 * first))

    x.grad.zero_()                          # the fix that every training loop uses
    (x**2).sum().backward()
    check("P6 zero_ resets", torch.allclose(x.grad, torch.tensor([2.0, 4.0])))


# =====================================================================
# PROBLEM 7 — backward() needs a SCALAR (and two ways to handle a vector)
# =====================================================================
def problem_7():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x**2                                 # a VECTOR output

    threw = False
    try:
        y.backward()                         # error: grad only implicit for scalars
    except RuntimeError:
        threw = True
    check("P7 vector.backward() errors", threw)

    # fix 1: reduce to a scalar first (the .sum() trick used for batches)
    x.grad = None
    (x**2).sum().backward()
    check("P7 fix via .sum()", torch.allclose(x.grad, 2 * x.detach()))

    # fix 2: tell backward how to weight each component
    x2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    (x2**2).backward(gradient=torch.ones_like(x2))
    check("P7 fix via gradient= arg", torch.allclose(x2.grad, 2 * x2.detach()))


# =====================================================================
# PROBLEM 8 — Functional grad vs .backward(); the graph is freed after use
# =====================================================================
def problem_8():
    r = torch.tensor([1.2], requires_grad=True)
    E = 4.0 * (r**-12 - r**-6)

    (g,) = torch.autograd.grad(E, r, retain_graph=True)   # returns grad, keeps graph
    E.backward()                                          # fills r.grad on same graph
    check("P8 autograd.grad == backward's .grad", torch.allclose(g, r.grad))

    # a fresh graph, backward twice without retaining -> error
    r2 = torch.tensor([1.2], requires_grad=True)
    E2 = 4.0 * (r2**-12 - r2**-6)
    E2.backward()
    threw = False
    try:
        E2.backward()                                     # graph already freed
    except RuntimeError:
        threw = True
    check("P8 second backward on freed graph errors", threw)


# =====================================================================
# PROBLEM 9 — Stopping the gradient: detach() and no_grad()
# =====================================================================
def problem_9():
    # detach() cuts one branch out of the graph:
    # y = x^2 + detach(x)^3  ->  dy/dx = 2x  (the cubed branch contributes 0)
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + x.detach()**3
    y.backward()
    check("P9 detached branch contributes 0", abs(x.grad.item() - 2 * 2.0) < 1e-12,
          f"x.grad = {x.grad.item()} (would be 16 if the cube were tracked)")

    # no_grad(): no graph is built at all
    with torch.no_grad():
        z = x * 5
    check("P9 no_grad -> no tracking", (z.requires_grad is False) and (z.grad_fn is None))


# =====================================================================
# PROBLEM 10 — backward fills .grad for a whole (N,3) tensor -> forces
# =====================================================================
def lj_energy(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    D = torch.linalg.norm(diff, dim=-1)
    iu = torch.triu_indices(pos.shape[0], pos.shape[0], offset=1)
    r = D[iu[0], iu[1]]
    return (4.0 * (r**-12 - r**-6)).sum()

def problem_10():
    pos = torch.tensor([[0.0, 0.0, 0.0],
                        [1.3, 0.0, 0.0],
                        [0.2, 1.2, 0.0],
                        [1.0, 1.0, 0.5]], requires_grad=True)
    E = lj_energy(pos)
    E.backward()
    forces = -pos.grad
    check("P10 .grad has the shape of its tensor (N,3)", tuple(forces.shape) == (4, 3))
    check("P10 net force ~ 0 (translational invariance)",
          torch.allclose(forces.sum(dim=0), torch.zeros(3), atol=1e-8),
          f"|net| = {forces.sum(dim=0).norm().item():.2e}")


if __name__ == "__main__":
    print("=" * 60)
    for i in range(1, 11):
        globals()[f"problem_{i}"]()
    print("=" * 60)
    print("ALL 10 AUTOGRAD PROBLEMS PASSED")
    print("=" * 60)