# Contributing

Thanks for helping. This guide covers setting up, the checks every change must
pass, and how changes and releases get made. [LEARNING.md](LEARNING.md)
explains *why* the project is built and checked the way it is.

## Set up

You need Python 3.11 or newer and git.

```bash
git clone https://github.com/MITRAKER/agentic-rag-assistant.git
cd agentic-rag-assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"     # the project plus its pinned dev tools
pre-commit install          # run the checks automatically on every commit
cp .env.example .env        # then add your ANTHROPIC_API_KEY if you use Claude
```

With `make` installed, `make install` does the `pip` and `pre-commit` steps,
and `make` alone lists every shortcut.

## The checks

CI runs all of these on every pull request, and a pull request can only be
merged when they pass. Run them locally first: it's faster than waiting for CI.

| Check | Command | `make` |
|-------|---------|--------|
| All pre-commit hooks: lint, format, types, Power of Ten, file hygiene | `pre-commit run --all-files` | |
| Lint | `ruff check .` | `make lint` |
| Format (also fixes what lint can fix) | `ruff format . && ruff check --fix .` | `make format` |
| Types, strict mode | `mypy` | `make typecheck` |
| NASA/JPL Power of Ten rules | `python tools/power_of_ten.py src tools` | `make power-of-ten` |
| Tests with branch coverage, at least 90% | `pytest --cov` | `make test` |
| Security: code and dependencies | `bandit -c pyproject.toml -r src tools` and `pip-audit --skip-editable` | `make security` |
| Everything above | | `make check` |

The tests run offline in a few seconds: they use an in-memory Qdrant, a
hashing embedder and a fake Claude client, so they need no API key, no
download and no Docker. Keep it that way: a test must never call a real API.

## Code standards

- **Style**: whatever `ruff format` produces. Don't argue with the formatter.
- **Types**: every function is fully annotated; `mypy --strict` must pass.
- **Power of Ten**: no recursion, no unbounded loops, functions of at most 60
  lines, preconditions checked with explicit `raise` (not `assert`), no
  mutable global state. The full list, adapted to Python, is in
  [LEARNING.md](LEARNING.md#engineering-standards).
- **Tests**: new behaviour comes with tests. A bug fix comes with a test that
  fails without the fix.
- **Docs**: update README.md, LEARNING.md and `.env.example` when behaviour or
  settings change, and add a line under "Unreleased" in CHANGELOG.md.
- **Secrets**: never commit an API key. `.env` is git-ignored for that reason.

## Commits and pull requests

1. Create a branch from `main`, for example `git switch -c fix/pdf-page-numbers`.
2. Make small, focused commits. Write messages in the
   [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
   style: `type: summary`, where the type is one of `feat`, `fix`, `docs`,
   `test`, `refactor`, `perf`, `build`, `ci` or `chore`. For example
   `fix: keep page numbers when a PDF page has no text`.
3. Push and open a pull request. Fill in the template's checklist.
4. CI runs. If a check fails, open its log, fix the cause, and push again.
5. A code owner reviews. When CI is green and the review is approved, the pull
   request is squash-merged, so `main` gets one clean commit per change.

## Releases

Versions follow [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.
While the version starts with `0.`, any release may change behaviour.

1. In a pull request, set `__version__` in `src/rag_assistant/__init__.py`, and
   in CHANGELOG.md move the "Unreleased" entries under a new heading such as
   `## [0.3.0] - 2026-12-01`. Update the links at the bottom of the file.
2. Merge it, then tag the merge commit and push the tag:

   ```bash
   git switch main && git pull
   git tag -a v0.3.0 -m "v0.3.0"
   git push origin v0.3.0
   ```

3. The Release workflow checks that the tag matches `__version__`, runs the
   tests, builds the package and publishes a GitHub release with the wheel, the
   source distribution and that version's CHANGELOG.md notes.

## Reporting problems

Use the issue forms for bugs and ideas. Report security problems privately, as
described in [SECURITY.md](SECURITY.md). Everyone taking part follows the
[code of conduct](CODE_OF_CONDUCT.md).
