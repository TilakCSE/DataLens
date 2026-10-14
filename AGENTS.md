# DataLens working agreements

- Read `Docs/core-contract.md` before changing the Python core.
- Keep the core deterministic and usable without an AI provider, database, or web service.
- Preserve `import datalens as dl` and `dl.analyse(df)` as the primary public interface.
- Reuse the same analysis report in the Python API and CLI.
- Keep analysis facts traceable to calculations on the supplied data. Do not invent confidence percentages or causal explanations.
- Add focused tests for behavior that could be wrong on real datasets, especially missing values, mixed types, and empty inputs.
- Do not edit the original project brief unless the user explicitly requests it.
- Before handing off work, run the relevant tests and state what passed or could not be run.
