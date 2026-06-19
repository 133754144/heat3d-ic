# Heat3D V4 Config Registry

Read this directory only for V4 registry, YAML generation, dry-run, or launch
readiness tasks.

Workflow:

1. Register the run in `run_registry.csv`.
2. Generate inherited YAML from `V4_base.yaml`; generated YAML stores only fields
   that differ from the base.
3. Resolve the inherited YAML in memory and run a dry-run command build.
4. Start tmux training only when the current user request explicitly approves a
   launch on a named server. Report the log path for live output.

The V4 standard task is:

```text
coords + k(x) + q(x) + BC -> T(x)
```
