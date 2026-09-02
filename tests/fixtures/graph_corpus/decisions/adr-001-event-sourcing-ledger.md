---
aliases: ["ADR-001", "event sourcing"]
tags: [decision]
---
# Adr 001 Event Sourcing Ledger

ADR-001 decides that [[ledger-service]] is event-sourced: every money movement is an immutable event and balances are projections. Rationale: auditability and replay after [[billing-service]] errors. Written by [[ana-lima]]; approved by [[team-payments]]. Alternative rejected: mutable balance rows with an audit table.
