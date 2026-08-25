---
name: spec-demo
description: >-
  Demonstrates full agentskills.io spec compliance. Use when testing
  multi-line YAML descriptions, bundled references, and scripts.
license: Apache-2.0
compatibility: Requires Python 3.11+
allowed-tools: read_skill_file run_skill_script
metadata:
  activation: on_demand
  version: "1.0.0"
tags: [testing, spec]
---

# Spec Demo Skill

Follow the workflow in `references/guide.md` (load it with read_skill_file).
Run `scripts/hello.py` to produce the greeting.
