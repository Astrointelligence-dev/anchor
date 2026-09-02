---
aliases: ["Catalog"]
tags: [service, discovery]
---
# Catalog Service

The Catalog service is the source of truth for products, prices and stock. Product updates are published to [[kafka-bus]] and indexed by [[search-service]]. Data lives in [[postgres-cluster]]. Owned by [[team-discovery]]; maintainer [[felipe-souza]]. Prices are stored in minor units (cents), never floats.
