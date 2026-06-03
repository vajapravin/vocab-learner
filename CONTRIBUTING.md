# Contributing

## Branching

- `main` is protected. Never commit or push directly to it.
- Every change lives on a short-lived branch off `main`.
- Branch naming: `<type>/<kebab-description>` where type is one of
  `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`.
- Delete branches after merge.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):

    <type>(<optional scope>): <imperative summary>

    <optional body>

    <optional footer>

Rules:

- Summary in imperative mood ("add", not "added" or "adds").
- Max 72 characters in the summary line.
- No trailing period on the summary.
- Body wraps at 100 characters, explains the *what* and *why*.

Examples:

    feat(extractor): add vision-based dictionary page extractor
    fix(llm): handle Anthropic APIError before JSON parse
    chore(git): add commit-msg hook enforcing Conventional Commits
    refactor(models): split VocabEntry sub-models into own module

## Pull Requests

- Open a PR from your feature branch into `main`.
- CI must be green (ruff, mypy, pytest) before merge.
- Self-review the diff before requesting review / merging.
- Squash-merge into `main`.
- Delete the source branch after merge.

## Local Checks Before Pushing

    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src
    uv run pytest

The `pre-push` git hook runs all of these automatically. To bypass in an
emergency: `git push --no-verify` (do not make this a habit).

## First Time After Cloning

Point git at the versioned hooks directory:

    git config core.hooksPath .githooks

Verify hooks are active:

    ls -la .githooks/
    git config --get core.hooksPath

## Working on a Change

    git checkout main
    git pull
    git checkout -b feat/your-change

    # ... edit, stage, commit ...
    git add -p                        # review each hunk as you stage
    git commit -m "feat(scope): summary"

    git push -u origin feat/your-change
    gh pr create --fill               # or open PR in browser

After PR is green and merged:

    git checkout main
    git pull
    git branch -d feat/your-change

## Secrets and Sensitive Data

- Never commit `.env`, API keys, tokens, or credentials.
- Never commit real user data or scanned dictionary pages containing personal notes.
- If you accidentally commit a secret: rotate it immediately, then rewrite history
  (`git filter-repo` or BFG) and force-push. Assume the secret is compromised.

## When in Doubt

Prefer smaller PRs. A 200-line PR gets reviewed properly; a 2000-line PR gets rubber-stamped.
