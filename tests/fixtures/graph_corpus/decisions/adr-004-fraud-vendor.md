---
aliases: ["ADR-004", "fraud vendor"]
tags: [decision]
---
# Adr 004 Fraud Vendor

ADR-004 selects the Sift vendor for [[fraud-scoring]] after a bake-off measuring precision at a fixed 1% false-positive rate. Evaluated by [[carla-mendes]]; approved by [[team-payments]]. Exit clause: a vendor SLA above 150 ms for a month triggers re-evaluation.
