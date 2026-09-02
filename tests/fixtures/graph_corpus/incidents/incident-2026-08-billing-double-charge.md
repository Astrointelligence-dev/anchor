---
aliases: ["double charge"]
tags: [incident]
---
# Incident 2026 08 Billing Double Charge

On 2026-08-09 [[billing-service]] issued duplicate invoices for 312 orders after a replayed settlement event from [[kafka-bus]] was not deduplicated. Detected by the [[ledger-service]] reconciliation job. Led by [[ana-lima]]; refunds completed the same day. Action item: idempotent event handling keyed by settlement id.
