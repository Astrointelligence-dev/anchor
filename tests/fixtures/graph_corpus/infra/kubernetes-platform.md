---
aliases: ["Kubernetes", "k8s"]
tags: [infra]
---
# Kubernetes Platform

All services run on the Kubernetes platform managed by [[team-platform]]: three clusters (dev, staging, prod) with autoscaling on CPU and queue depth. Secrets are injected from [[vault-secrets]]. Deployments are canary by default with automatic rollback on error-rate alerts from [[observability-stack]].
