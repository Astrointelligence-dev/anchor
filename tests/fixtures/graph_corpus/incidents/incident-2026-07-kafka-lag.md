---
aliases: ["Kafka lag incident"]
tags: [incident]
---
# Incident 2026 07 Kafka Lag

On 2026-07-22 consumer lag on [[kafka-bus]] reached 4 hours for [[notification-service]] because a partition rebalance storm followed a broker upgrade. Detected by the lag alert in [[observability-stack]]; handled by [[henrique-alves]] using the [[oncall-runbook]]. Order confirmations were delayed but not lost.
