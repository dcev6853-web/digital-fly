# Digital Fly — Mac Setup Guide

## Prerequisites

- **macOS** (Intel or Apple Silicon — both work, a couple steps differ, noted below)
- **Xcode Command Line Tools** (needed to build some Python packages)
  ```bash
  xcode-select --install
  ```
- **Homebrew** (if you don't have it)
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
- **Python 3.10 or 3.11** (check first, FlyGym doesn't support the very latest Python yet)
  ```bash
  python3 --version
  ```
  If you need to install/switch:
  ```bash
  brew install python@3.11
  ```
- **Git**
  ```bash
  git --version   # ships with Xcode CLT above
  ```
- **GitHub CLI** (optional but easiest for auth)
  ```bash
  brew install gh
  gh auth login
  ```
- **A repo invite from Henry** — you need to accept the collaborator invite email before cloning.

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/dcev6853-web/digital-fly.git
cd digital-fly

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

If `pip install` fails on a package needing compilation (common on Apple Silicon for some
scientific packages), try:
```bash
arch -arm64 pip install -r requirements.txt
```
If a specific package still fails, tell Henry the exact error — some deps may need a
Rosetta fallback on M-series chips.

## Running it

Mac needs a different rendering backend than Linux (`egl` doesn't exist on Mac):

```bash
# try this first
MUJOCO_GL=glfw .venv/bin/python experiments/evaluate.py --run experiments/results/manc_v2 --episodes 3 --visual

# if that errors, fall back to:
MUJOCO_GL=osmesa .venv/bin/python experiments/evaluate.py --run experiments/results/manc_v2 --episodes 3 --visual
```

This renders a fresh episode (a couple minutes) and opens/saves a video showing the
connectome-wired brain navigating to a target.

## Sanity check it's working

```bash
.venv/bin/pytest -q
```
Should show all tests passing. If this fails before you've changed anything, something's
wrong with the environment setup, not the project — send Henry the output.

## Quick reference — other things you can run

```bash
# watch a different brain
MUJOCO_GL=glfw .venv/bin/python experiments/evaluate.py --run experiments/results/baseline_mlp --episodes 3 --visual

# check current project status/results
cat experiments/results/report.md
```

## If something breaks

Send Henry: the exact command you ran, the full error output, and `python3 --version` /
`uname -m` (tells us Intel vs Apple Silicon).
## Installing Pip
cd digital-fly

# confirm python3 exists
python3 --version

# recreate the venv (ensurepip bootstraps pip automatically)
python3.11 -m venv .venv --upgrade-deps

# activate it
source .venv/bin/activate

# confirm pip now exists inside the venv
which pip
pip --version
