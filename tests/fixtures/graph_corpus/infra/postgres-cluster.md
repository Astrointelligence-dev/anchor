---
aliases: ["Postgres", "PostgreSQL"]
tags: [infra, database]
---
# Postgres Cluster

The Postgres cluster is a three-node PostgreSQL 17 setup with synchronous replication, used by [[payments-service]], [[ledger-service]], [[user-profile]] and [[catalog-service]]. Backups run hourly with 30-day retention. Managed by [[team-platform]] on [[kubernetes-platform]]. Connection pooling via PgBouncer.
