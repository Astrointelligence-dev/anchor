---
aliases: ["Billing"]
tags: [service, payments]
---
# Billing Service

Billing turns settled payments into invoices and receipts, and records every charge in [[ledger-service]] following [[adr-001-event-sourcing-ledger]]. It reads settlement events from [[kafka-bus]]. Owned by [[team-payments]]; on-call [[ana-lima]]. The duplicate invoices are documented in [[incident-2026-08-billing-double-charge]]. Invoices are immutable once issued.
