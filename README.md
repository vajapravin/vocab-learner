
## Development Workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

**First time after cloning:**

```bash
git config core.hooksPath .githooks
```

**Every change:**

```bash
git checkout main
git pull
git checkout -b feat/your-change
# ... edit, commit, push ...
gh pr create   # or open PR in browser
# after CI green and self-review: squash-merge, delete branch
```
