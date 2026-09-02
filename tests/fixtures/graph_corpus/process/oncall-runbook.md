---
aliases: ["runbook", "on-call runbook"]
tags: [process]
---
# Oncall Runbook

The on-call runbook lists the first steps for every alert: check dashboards in [[observability-stack]], confirm [[kubernetes-platform]] rollout status, inspect [[kafka-bus]] consumer lag, and page the owning team per [[adr-005-oncall-rotation]]. Maintained by [[henrique-alves]]. Severity 1 requires a status page update within 15 minutes.
