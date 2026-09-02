---
aliases: ["Fraud Scoring", "fraud"]
tags: [service, risk]
---
# Fraud Scoring

Fraud Scoring assigns a risk score from 0 to 100 to every order using the vendor chosen in [[adr-004-fraud-vendor]]. Scores above 80 block the authorization in [[checkout-service]]. The model features come from [[user-profile]] purchase history and device fingerprints. Maintained by [[carla-mendes]] on [[team-payments]]. The vendor SLA is 150 ms.
