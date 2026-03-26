---
name: review-pull-request
description: Review GitHub pull requests and local diffs for bugs, regressions, API mismatches, missing tests, and unsafe assumptions. Use when the user asks to review a PR, inspect a diff line by line, prioritize findings by severity, try a small reproducible example, or comment on the PR directly.
---

# Review Pull Request

Review the diff as a correctness task first, not a summarization task. Favor concrete bugs, regressions, and missing validation over style comments.

## Workflow

1. Load PR context. Fetch the PR title, body, commits, touched files, and full diff. For a local review request, gather the equivalent git diff and file list.
2. Read every changed file line by line. Open the surrounding code so the review is grounded in the real contract, not only the patch hunk.
3. Trace downstream effects. Check imports, exports, public APIs, schemas, CSV fields, parameter names, shapes, and callers that consume the changed behavior.
4. Look for concrete breakage first. Prioritize correctness bugs, behavioral regressions, shape mismatches, silent data loss, missing validation, and tests that do not prove the claimed behavior.
5. Run a quick check. Prefer targeted tests or a disposable script that finishes in under one minute. If a full integration run is too slow, exercise the narrowest changed path that can falsify the PR claim.
6. Label findings by importance. Use `[P0]` for release-blocking breakage or data loss, `[P1]` for high-confidence correctness bugs, `[P2]` for meaningful but non-blocking issues, and `[P3]` for polish or minor reporting issues.
7. Comment PR directly when requested. Use `gh pr review` for summary comments and `gh api` for file-specific comments when line targeting matters. Keep each comment focused on one issue, explain the impact, and suggest the smallest safe fix.
8. Report residual risk. If no findings are found, say so explicitly and mention any gaps such as unrun tests, unverified integrations, or assumptions made from limited local context.

## Review Heuristics

- Compare the PR summary against what the code actually does.
- Verify that tests cover the exact behavior the PR claims to add or fix.
- Treat new exports as API changes and check dependencies, docs, and downstream call sites.
- Check naming and shape conventions whenever arrays, records, CSVs, metrics, or schema-aware outputs change.
- For condition-aware or hierarchical data, verify every level uses the same key format.
- Prefer reproducing a failure with a tiny example over speculating.

## Minimal Example

- Use a short disposable script or a narrowly scoped test command.
- Print or summarize the observed output so the example supports the review conclusion.
- Do not save the example into the repo.

## Output Format

- Present findings first, ordered by severity.
- Include file and line references.
- Keep summaries brief and center the failure mode, impact, and missing coverage.
