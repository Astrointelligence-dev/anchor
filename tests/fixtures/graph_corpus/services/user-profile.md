---
aliases: ["Profile", "User Profile"]
tags: [service, identity]
---
# User Profile

User Profile stores names, addresses and preferences in [[postgres-cluster]], and exposes purchase history that [[fraud-scoring]] consumes. Personal data is encrypted at rest with keys from [[vault-secrets]]. Owned by [[team-identity]]; maintainer [[elisa-nunes]]. Address validation calls an external geocoder with a 2-second timeout.
