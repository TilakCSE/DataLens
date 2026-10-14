# DataLens core agent workflow

The local Git repository has a `main` branch connected to `https://github.com/TilakCSE/DataLens.git`. The core work is split into three isolated Git worktrees so agents can edit separate files concurrently.

| Branch | Task | Owned files |
| --- | --- | --- |
| `feature/core-ingestion` | CSV and Excel loading | `src/datalens/io.py`, `tests/test_io.py` |
| `feature/core-analysis` | Profiling and data quality | `src/datalens/analysis.py`, `tests/test_analysis.py` |
| `feature/core-demo` | Chart recommendations and CLI demo | `src/datalens/recommend.py`, `src/datalens/cli.py`, related tests, `examples/`, `README.md` |

The coordinator owns the shared `src/datalens/models.py`, `src/datalens/api.py`, package exports, integration tests, and final assembly. Shared contracts are in `Docs/core-contract.md`.

## Review and integration

1. Inspect each agent's test results and branch diff: `git diff main...feature/core-ingestion`, and similarly for the other branches.
2. Request corrections on the agent branch until its acceptance criteria are met.
3. Merge completed branches into `main` one at a time. Resolve any integration issues in `main` while preserving each component's tests.
4. Run the full test suite, execute both example datasets through the CLI, and confirm the JSON output parses.
5. Review the combined diff and demo output before pushing `main` to GitHub.
6. Keep the branches and worktrees until the work is accepted. Then remove or archive the temporary worktrees using the Codex worktree controls.

Useful local checks:

```powershell
git worktree list
git branch --all
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\datalens.exe analyse examples\sales.csv
```

The `.venv` folder belongs to the local checkout and is ignored by Git. A newly created worktree needs its own environment or an explicitly selected Python interpreter. Worktree branches can be reviewed and merged locally while GitHub is unavailable.
