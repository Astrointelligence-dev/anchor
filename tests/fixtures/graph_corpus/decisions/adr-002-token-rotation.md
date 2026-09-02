---
aliases: ["ADR-002", "token rotation"]
tags: [decision]
---
# Adr 002 Token Rotation

ADR-002 mandates refresh-token rotation in [[auth-service]] and 90-day rotation of signing keys in [[vault-secrets]], motivated by [[incident-2026-05-auth-token-leak]]. Author [[diego-rocha]]. Rejected alternative: longer-lived opaque tokens with a revocation list.
