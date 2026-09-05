# ASE Setup — Windows + VS Code + Conda

A clean, isolated environment for ASE + MACE work. Tested on Windows 10/11 with
Anaconda, VS Code, ASE 3.29.0, and mace-torch.

Do the **one-time setup** once. After that you only ever need the
**daily workflow** at the bottom.

---

## One-time setup

### 0. Update conda first

Anaconda installers often ship a conda version that is **incompatible with
PowerShell 7.5 and newer** — every conda command fails with an empty-string
argument error. Fixing this first avoids the single most confusing failure
mode in this whole setup. Open **Anaconda Prompt** (Start menu) and run:

```
conda update -n base -c conda-forge "conda>=25" -y
```

> Requesting `"conda>=25"` explicitly matters — just running `conda update
> conda` can silently resolve to the same old version if something in `base`
> is holding it back. Asking for a version floor forces the solver to
> actually move forward.

Check it worked:

```
conda --version
```

You want **25.0 or higher**. If it's still below that, re-run the update
command once more — repeat runs are harmless.

### 1. Open Anaconda Prompt

Search "Anaconda Prompt" in the Start menu.

Use this, not `cmd` or PowerShell — `conda` is not on the Windows PATH by default.

### 2. Create the environment

```
conda create -n asemd python=3.11 -y
```

> **Never install into `base`.** Keeping ASE in its own environment means a broken
> package can be fixed by deleting one environment instead of reinstalling Anaconda.

### 3. Activate it

```
conda activate asemd
```

Your prompt changes from `(base)` to `(asemd)`. That prefix is how you always know
which environment you are in.

### 4. Install the packages

```
pip install ase==3.29.0 numpy matplotlib jupyterlab ipykernel mace-torch
```

Pinning ASE's version matters: it renamed several molecular-dynamics functions
in 3.28 and 3.29, so an unpinned install can silently break older scripts.
`mace-torch` pulls in a CPU build of PyTorch automatically — it's the largest
download in this whole setup, so expect it to take a couple of minutes.

> **Need GPU/CUDA acceleration?** Install a CUDA-matched build of `torch`
> yourself first (see [pytorch.org](https://pytorch.org/get-started/locally/)
> for the right command for your CUDA version), then run the line above —
> pip will detect torch is already present and won't overwrite it with the
> CPU build.

### 5. Verify the install

```
python -c "import ase; print(ase.__version__)"
```

Expected output:

```
3.29.0
```

Then check MACE imports cleanly (this is what broke last time — catch it here
instead of mid-script):

```
python -c "from mace.calculators import mace_mp; print('mace OK')"
```

Then check the GUI works — this is the one piece that can fail silently on Windows:

```
python -c "from ase.build import bulk; from ase.visualize import view; view(bulk('Cu','fcc',a=3.61,cubic=True)*(2,2,2))"
```

A window with 32 copper atoms should open. Close it to continue.

### 6. Register the Jupyter kernel

```
python -m ipykernel install --user --name asemd --display-name "ASE MD"
```

This makes the environment selectable as a kernel inside VS Code notebooks.

### 7. Let PowerShell see conda

Still in **Anaconda Prompt**:

```
conda init powershell
```

Then close VS Code completely and reopen it.

> Running `conda init powershell` from inside PowerShell fails with
> `conda: The term 'conda' is not recognized` — that is the exact problem this
> command fixes, so it has to be run from Anaconda Prompt. Doing Step 0 first
> means this hook will actually work once PowerShell picks it up — on the old
> conda version, the hook itself is what carries the PowerShell-7.5 bug.

### 8. Point VS Code at the environment

1. Install the **Python** and **Jupyter** extensions (both by Microsoft).
2. `File → Open Folder` → your ASE project folder.
3. `Ctrl+Shift+P` → type `Python: Select Interpreter` → choose the entry whose path
   contains `envs\asemd`.
4. For notebooks, use the kernel picker at the **top right of the notebook**
   and select **ASE MD**.

The interpreter choice is saved per folder in `.vscode/settings.json`. This
only affects notebooks and any UI that reads the selected interpreter — it
does **not** control which environment a terminal activates. See below.

---

## Daily workflow

If you want to freely pick *any* environment after opening VS Code, regardless
of which folder is open, skip the interpreter picker entirely for terminal
work and just activate manually:

1. Open VS Code
2. Open a terminal — `Ctrl+` `` ` ``
3. Run:
   ```
   conda activate asemd
   ```
4. Confirm the prompt shows `(asemd)` and start coding

This works in any terminal, in any folder, regardless of VS Code's interpreter
selection — as long as Steps 0 and 7 above were done once. You do not need to
open Anaconda Prompt again unless you are creating a new environment.

You only need the interpreter/kernel picker (Step 8) for **notebooks**, since
notebook cells use the kernel picker rather than the terminal's active
environment.

Sanity check whenever you are unsure which Python is active:

```powershell
python -c "import sys, ase; print(sys.prefix); print(ase.__version__)"
```

The printed path should contain `envs\asemd`.

---

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `Invoke-Expression: Cannot bind argument to parameter 'Command' because it is an empty string` on any conda command | Conda older than 25.0 is incompatible with PowerShell 7.5+ | From Anaconda Prompt: `conda update -n base -c conda-forge "conda>=25" -y`, then open a fresh terminal |
| `conda: The term 'conda' is not recognized` | PowerShell has not been initialised | Run `conda init powershell` from **Anaconda Prompt**, restart VS Code |
| `profile.ps1 cannot be loaded because running scripts is disabled` | Windows blocks profile scripts by default | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then reopen the terminal |
| `ModuleNotFoundError: No module named 'ase'` | Terminal and editor are using different Pythons | Run `conda activate asemd` in the terminal directly (see Daily workflow) rather than relying on the interpreter picker |
| `ModuleNotFoundError: No module named 'mace'` | `mace-torch` was never installed in this environment | `pip install mace-torch` inside the activated environment |
| Notebook cannot find ASE or MACE | Wrong kernel selected | Kernel picker at the top right of the notebook → **ASE MD** |
| `view(atoms)` opens nothing | tkinter missing or blocked | Reinstall the environment with `python=3.11` from Anaconda's own Python, which ships tkinter |
| Terminal always opens in `(base)` | Either auto-activation is off, **or** you just want manual control | Enable `python.terminal.activateEnvironment` in VS Code settings if you want auto-activation — or just run `conda activate asemd` yourself each time (recommended if you switch environments often) |

---

## A note on project location

Avoid putting the project inside a OneDrive-synced folder. Trajectory files get
large, and OneDrive will try to sync every `.traj` as it is written — this slows
runs and can lock files mid-write.

Keep the repository somewhere local, for example `C:\Users\<you>\ASE_tutorials`,
and use Git for backup and history instead.

---

## Rebuilding this environment elsewhere

> **Gap in the original guide:** `ase`, `mace-torch`, `numpy`, `matplotlib`,
> `jupyterlab`, and `ipykernel` were all installed with `pip`, not `conda
> install`. `conda env export --from-history` only records packages installed
> *via conda* — every one of those pip packages is silently missing from that
> file. Following the original two-command rebuild would recreate an empty
> Python 3.11 environment with none of the actual tools in it.

On the source machine, export **both** the conda-level environment and the
pip-level packages:

```
conda env export --from-history > environment.yml
pip freeze > requirements.txt
```

On the target machine, recreate the environment, then install the pip layer
on top:

```
conda env create -f environment.yml
conda activate asemd
pip install -r requirements.txt
```

`--from-history` keeps `environment.yml` portable across operating systems by
recording only what you explicitly asked conda for; `requirements.txt` covers
everything else.
