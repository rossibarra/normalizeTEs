# Repository instructions

- Ask before modifying any existing non-code/data file, and make a `.bak` copy after permission is granted.
- Before modifying any target file, check it for uncommitted changes; if present, offer the standard commit/backup choices.
- Never edit through a symlink. Replace it safely with a regular file and record the original link in `symlinks.md`.
- Keep project writes inside this repository; temporary working files may go under `/tmp`.
- If this repository lacks `AGENTS.md`, add it before other work.
- Before Python, pytest, pip, or project CLI commands, initialize conda in a login Bash shell. If `environment.yml` exists, activate the environment named there and stop if activation fails.

When confirmation is required for an existing data file, offer exactly:

1. commit, backup, and modify
2. don't commit, but backup and modify
3. do not modify
4. other
