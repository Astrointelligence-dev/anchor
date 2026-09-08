---
aliases: ["Notifications"]
tags: [service, platform]
---
# Notification Service

Notifications sends email, SMS and push messages triggered by events on [[kafka-bus]]: order confirmations from [[checkout-service]], receipts from [[billing-service]]. Templates are versioned in git. Owned by [[team-platform]]; maintainer [[henrique-alves]]. Delivery retries back off exponentially up to 6 hours.
