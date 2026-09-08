---
aliases: ["Ledger"]
tags: [service, payments]
---
# Ledger Service

The Ledger is an append-only, double-entry book of all money movements, event-sourced per [[adr-001-event-sourcing-ledger]] and stored in [[postgres-cluster]]. Balances are projections rebuilt from events. Owned by [[team-payments]]; maintainer [[bruno-costa]]. A daily reconciliation job compares the ledger with the acquirer statement.
