# Security policy

## Supported versions

The project is young, so only the latest release receives security fixes.

| Version | Supported |
|---------|-----------|
| 0.2.x   | ✅        |
| < 0.2   | ❌        |

## Reporting a vulnerability

Please **don't open a public issue** for a security problem. Report it
privately instead:

1. Open the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Describe the problem, how to reproduce it, and what an attacker could do with it.

You'll get a first reply within a week. When a fix is released, the advisory
is published, and you're credited in it if you'd like to be.

## What counts as a vulnerability

For example:

- Document content that makes the assistant reveal the API key, read files
  outside the folder it was asked to ingest, or run commands. (Prompt injection
  that only changes the wording of an answer is a quality bug: please open a
  normal issue.)
- A crafted PDF, HTML or Markdown file that crashes ingestion or exhausts
  memory in spite of the file-size and file-count limits.
- A dependency with a known vulnerability that our checks missed.

Problems that need an attacker who already controls your computer or your
`.env` file are out of scope.

## How the repository protects itself

- **Secrets**: the API key is read from the environment or a git-ignored `.env`
  file and kept in a `SecretStr`, so it never appears in logs. A pre-commit
  hook blocks committed private keys.
- **Code scanning**: CodeQL and Bandit check every pull request, and run again
  every week.
- **Dependencies**: pip-audit checks every pull request and runs every week,
  and Dependabot opens pull requests to update dependencies.
- **CI permissions**: workflows can only read the repository. The release job is
  the one exception: it may create releases.
