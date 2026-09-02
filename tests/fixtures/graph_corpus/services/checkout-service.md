---
aliases: ["Checkout"]
tags: [service, payments]
---
# Checkout Service

The Checkout service assembles the cart, computes taxes and hands the card authorization to [[payments-service]]. Every order is scored by [[fraud-scoring]] before authorization. Checkout runs on [[kubernetes-platform]] and keeps cart state in [[redis-cache]]. Owned by [[team-payments]]; the tech lead is [[ana-lima]]. Latency budget: 300 ms p95. Feature flags are read from the config map, never from environment variables.
