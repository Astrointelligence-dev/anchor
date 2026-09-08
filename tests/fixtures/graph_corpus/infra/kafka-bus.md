---
aliases: ["Kafka", "event bus"]
tags: [infra, messaging]
---
# Kafka Bus

Kafka is the event bus: settlement events from [[payments-service]], product updates from [[catalog-service]], and every notification trigger for [[notification-service]]. Consumer lag alerts are defined in [[observability-stack]]. The July lag incident is [[incident-2026-07-kafka-lag]]. Retention is 7 days; topics are partitioned by entity id.
