---
aliases: ["Payments"]
tags: [service, payments]
---
# Payments Service

The Payments service talks to the card acquirer, stores authorizations in [[postgres-cluster]] and publishes settlement events to [[kafka-bus]]. Retries use idempotency keys derived from the order id. On-call engineer: [[bruno-costa]]; escalation goes to [[team-payments]]. The March outage is documented in [[incident-2026-03-payments-outage]]. Settlement batches close at 23:00 UTC.
