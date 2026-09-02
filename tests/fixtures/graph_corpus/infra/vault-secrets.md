---
aliases: ["Vault"]
tags: [infra, security]
---
# Vault Secrets

Vault stores signing keys for [[auth-service]], encryption keys for [[user-profile]] and acquirer credentials for [[payments-service]]. Keys rotate every 90 days per [[adr-002-token-rotation]]. Administered by [[diego-rocha]]. Access is audited and reviewed quarterly by [[team-identity]].
