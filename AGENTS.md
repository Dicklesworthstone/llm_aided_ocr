# AGENTS.md

## Permanent CI Policy — DSR Only

**Never use GitHub Actions for ANY purpose in this project.** This includes GitHub-hosted
runners and self-hosted runners registered with GitHub Actions, CI, tests, builds, releases,
deployments, scheduled jobs, and maintenance. Do not enable, dispatch, rerun, or rely on
Actions, and do not treat billing or runner availability as a reason to restore it.

Use **dsr ([Dicklesworthstone/doodlestein_self_releaser](https://github.com/Dicklesworthstone/doodlestein_self_releaser))**
directly for these workloads instead, not merely as an Actions fallback. Existing workflow
YAML may be retained as input for local dsr/act execution; that does not authorize execution
by GitHub Actions. Preserve the workload definitions and their required checks.

