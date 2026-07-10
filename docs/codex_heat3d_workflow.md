# Heat3D Remote Server Connection Workflow

This document covers only SSH connection, remote repository alignment, and
remote Python-environment checks for Heat3D servers.

## Quick Contract

- Read this section before using a Heat3D server, then read
  `WF-REMOTE-CONNECT` for the connection procedure.
- Use the SSH aliases `devbox` or `wsl2`; do not substitute an unverified host.
- The repository remote must use SSH:
  `git@github.com:133754144/heat3d-ic.git`.
- Confirm the remote branch, commit, and worktree status before relying on a
  server checkout.
- Activate the `rigno` conda environment before running project Python
  commands.

## WF-REMOTE-CONNECT: SSH, Repository, And Environment Check

Connect to the target server:

```bash
ssh devbox
# Or: ssh wsl2
```

On the server, verify the checkout before updating it. Do not pull when
`git status --short` reports changes that you do not intend to keep.

```bash
cd ~/myCodeGitOnly/heat3d-ic
git remote get-url origin
git branch --show-current
git rev-parse --short HEAD
git status --short
```

For the V5 checkout, after confirming that the worktree is clean, align it with
GitHub:

```bash
git fetch origin
git switch research/v5
git pull --ff-only origin research/v5
git branch --show-current
git rev-parse --short HEAD
git status --short
```

Initialize and verify the Heat3D Python environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rigno
python --version
python -c "import jax; print(jax.devices())"
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not in PATH"
```

For a non-interactive SSH command, initialize conda in the same remote shell:

```bash
ssh devbox 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate rigno && cd ~/myCodeGitOnly/heat3d-ic && python --version'
```

## Connection Troubleshooting

- `ssh devbox` or `ssh wsl2` fails: verify the corresponding host alias, SSH
  key, and network access in `~/.ssh/config`; do not guess another server.
- `conda: command not found`: source `~/miniconda3/etc/profile.d/conda.sh`
  before `conda activate rigno`.
- Project imports or Python packages are missing: verify that `rigno` is active
  and use `python` from that environment rather than assuming `python3` exists.
- The server shows an unexpected branch or commit: inspect `git status --short`
  first, then use the clean-worktree update sequence above.
- `jax.devices()` has no GPU: this can be normal on WSL-backed servers; record
  the result and do not infer CUDA availability from `nvidia-smi` alone.
