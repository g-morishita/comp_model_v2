# AGENTS

## Hard Rule
- No backward compatibility.
- Do not add aliases, deprecated paths, migration shims, dual schema support, or compatibility wrappers.
- If an API changes, update all call sites, tests, and docs to the new API immediately.
- Only add compatibility if explicitly requested in that exact task.

## DocString
- Please add docstrings to every class, functions in Numpy-Style
- For each phase, please make a branch, commit, and make pull request
