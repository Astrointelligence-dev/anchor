---
aliases: ["Auth"]
tags: [service, identity]
---
# Auth Service

The Auth service issues JWT access tokens with a 15-minute lifetime and refresh tokens rotated per [[adr-002-token-rotation]]. Signing keys live in [[vault-secrets]]. Sessions are stored in [[redis-cache]]. Owned by [[team-identity]]; tech lead [[diego-rocha]]. The token leak is documented in [[incident-2026-05-auth-token-leak]].
