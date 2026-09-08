---
aliases: ["March payments outage"]
tags: [incident]
---
# Incident 2026 03 Payments Outage

On 2026-03-12 [[payments-service]] was down for 47 minutes after a failover of [[postgres-cluster]] left the connection pool pointing at the old primary. Orders queued in [[checkout-service]] and were authorized after recovery. Postmortem by [[bruno-costa]]. Action item: health checks now verify write capability.
