---
aliases: ["Search"]
tags: [service, discovery]
---
# Search Service

Search indexes the catalog into the engine selected in [[adr-003-search-engine]] and serves autocomplete and faceted queries. It consumes product events from [[kafka-bus]]. Owned by [[team-discovery]]; on-call [[gabriela-torres]]. The June latency regression is in [[incident-2026-06-search-latency]]. Index rebuilds run nightly at 02:00 UTC.
