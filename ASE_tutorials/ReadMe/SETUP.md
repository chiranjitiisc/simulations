# ASE Setup — Windows + VS Code + Conda

A clean, isolated environment for ASE work. Tested on Windows 10/11 with Anaconda,
VS Code, and ASE 3.29.0.

Do the **one-time setup** once. After that you only ever need the
**daily workflow** at the bottom.

---

## One-time setup

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
pip install ase==3.29.0 numpy matplotlib jupyterlab ipykernel
```

Pinning the version matters: ASE renamed several molecular-dynamics functions in
3.28 and 3.29, so an unpinned install can silently break older scripts.

### 5. Verify the install

```
python -c "import ase; print(ase.__version__)"
```

Expected output:

```
3.29.0
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
> command fixes, so it has to be run from Anaconda Prompt.

### 8. Point VS Code at the environment

1. Install the **Python** and **Jupyter** extensions (both by Microsoft).
2. `File → Open Folder` → your ASE project folder.
3. `Ctrl+Shift+P` → type `Python: Select Interpreter` → choose the entry whose path
   contains `envs\asemd`.
4. For notebooks, use the kernel picker at the **top right of the notebook** and
   select **ASE MD**.

The interpreter choice is saved per folder in `.vscode/settings.json`, so it sticks
across restarts and does not affect your other projects.

---

## Daily workflow

Setup is done. From now on:

1. Open VS Code
2. Open your ASE project folder
3. Open a terminal — `Ctrl+` `` ` ``
4. Confirm it shows `(asemd)` and start coding

That's it. **You do not need to open Anaconda Prompt again** unless you are creating
a new environment.

Sanity check whenever you are unsure which Python is active:

```powershell
python -c "import sys, ase; print(sys.prefix); print(ase.__version__)"
```

The printed path should contain `envs\asemd`.

---

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `conda: The term 'conda' is not recognized` | PowerShell has not been initialised | Run `conda init powershell` from **Anaconda Prompt**, restart VS Code |
| `profile.ps1 cannot be loaded because running scripts is disabled` | Windows blocks profile scripts by default | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then reopen the terminal |
| `ModuleNotFoundError: No module named 'ase'` | Terminal and editor are using different Pythons | Re-run `Python: Select Interpreter` and pick `asemd`; confirm the terminal shows `(asemd)` |
| Notebook cannot find ASE | Wrong kernel selected | Kernel picker at the top right of the notebook → **ASE MD** |
| `view(atoms)` opens nothing | tkinter missing or blocked | Reinstall the environment with `python=3.11` from Anaconda's own Python, which ships tkinter |
| Terminal opens in `(base)` | Auto-activation off | Enable `python.terminal.activateEnvironment` in VS Code settings |

---

## A note on project location

Avoid putting the project inside a OneDrive-synced folder. Trajectory files get
large, and OneDrive will try to sync every `.traj` as it is written — this slows
runs and can lock files mid-write.

Keep the repository somewhere local, for example `C:\Users\<you>\ASE_tutorials`,
and use Git for backup and history instead.

---

## Rebuilding this environment elsewhere

To reproduce the environment on another machine:

```
conda env export --from-history > environment.yml
```

And on the target machine:

```
conda env create -f environment.yml
conda activate asemd
```

Using `--from-history` records only the packages you explicitly asked for, which
keeps the file portable across operating systems.
