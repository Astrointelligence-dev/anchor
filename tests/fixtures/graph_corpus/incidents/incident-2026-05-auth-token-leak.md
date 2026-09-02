---
aliases: ["token leak"]
tags: [incident]
---
# Incident 2026 05 Auth Token Leak

On 2026-05-04 a refresh token from [[auth-service]] appeared in application logs shipped to the [[observability-stack]]. No misuse was detected. Response led by [[diego-rocha]]; the outcome is [[adr-002-token-rotation]] and log redaction rules. All refresh tokens were revoked within 2 hours.
