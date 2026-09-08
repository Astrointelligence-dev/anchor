---
aliases: ["Redis"]
tags: [infra, cache]
---
# Redis Cache

Redis provides session storage for [[auth-service]] and cart state for [[checkout-service]]. Keys expire after 24 hours. Runs as a six-node cluster managed by [[team-platform]]. Persistence is disabled; the cache is rebuildable.
