---
name: git-workflow
description: Git branch strategy — develop as operations branch, upstream sync, and PR workflow.
---

# Git Workflow

## Remotes

| Remote     | Purpose                        | URL                                              |
|------------|--------------------------------|--------------------------------------------------|
| `origin`   | Personal fork (private)        | `github-codex:codestrongestx/parameter-golf.git` |
| `upstream` | Public repo (openai)           | `https://github.com/openai/parameter-golf.git`   |

## Branches

| Branch    | Tracks          | Purpose                                                        |
|-----------|-----------------|----------------------------------------------------------------|
| `main`    | `upstream/main` | Mirror of the public repo. Keep clean — never commit directly. |
| `develop` | `origin/develop`| Operations branch. All private work lives here: skills, experiment notes, run logs, grant docs. |

Feature branches are cut **only** for submitting PRs upstream.

## Common Operations

### Sync `records/` (or any folder) from upstream into develop

```bash
git fetch upstream main
git checkout upstream/main -- records
git commit -m "sync(records): pull latest leaderboard records from upstream"
git push origin develop
```

### Sync all of main into develop

```bash
git fetch upstream main
git merge upstream/main --no-edit
# resolve conflicts if any (prefer ours for private files, theirs for upstream code)
git push origin develop
```

### Prepare a PR branch

```bash
# Start from latest upstream main, not develop
git fetch upstream main
git checkout -b my-pr-branch upstream/main

# Do the work, commit, then push to origin
git push origin my-pr-branch -u

# Open PR against upstream
gh pr create --repo openai/parameter-golf --base main
```

**Important:** PR branches must be based on `upstream/main`, not `develop`. The `develop` branch contains private files (skills, experiment notes) that should not go upstream.

### After PR is merged

```bash
# Update main and develop
git fetch upstream main
git branch -f main upstream/main
git checkout develop
git merge main --no-edit
git push origin develop
```
