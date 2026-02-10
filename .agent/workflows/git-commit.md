---
description: How to handle git commits that require GPG signing
---

## Git Commit Rules

1. **Never bypass GPG signing.** Do not use `--no-gpg-sign` or `-S` flags to work around signing failures.
2. If a `git commit` fails due to GPG/pinentry issues (e.g. "Screen or window too small"), **notify the user** and ask them to run the commit manually in their own terminal.
3. Suggest the exact commit command and message so the user can copy-paste it.
4. Generally, any time a command requires user-side interaction (passphrase, authentication, confirmation), do NOT try to bypass it — notify the user instead.
