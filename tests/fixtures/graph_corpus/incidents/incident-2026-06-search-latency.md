---
aliases: ["search latency regression"]
tags: [incident]
---
# Incident 2026 06 Search Latency

On 2026-06-18 [[search-service]] p95 latency rose from 80 ms to 900 ms after a schema change in [[catalog-service]] doubled the index size. Rolled back within 40 minutes by [[gabriela-torres]]. Action item: an index size budget enforced in CI.
