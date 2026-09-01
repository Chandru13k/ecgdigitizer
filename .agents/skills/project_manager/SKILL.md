---
name: project_manager
description: Tracks progress, controls task.md lists, prevents scope creep, and coordinates timeline goals.
---

# Project Manager Skill

This skill governs planning and task management to ensure execution alignment.

## 📋 Objective
To maintain the `task.md` workflow, flag deviations from goals, and organize the task lists into sequential components.

## ⚙️ Workflow & Steps
1. **Initialize Project State**: Read user guidelines and create the initial `task.md` list.
2. **Track Task Transition**: Mark tasks as `[ ]`, `[/]`, or `[x]` as work advances.
3. **Prevent Scope Creep**: Compare proposed changes against the approved implementation plan. Flag any unapproved changes to the Master Orchestrator.
4. **Transition Gate**: Confirm that all verification checks are passed before allowing the task to be marked as complete.
