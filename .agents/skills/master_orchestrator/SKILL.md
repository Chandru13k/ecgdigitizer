---
name: master_orchestrator
description: Plans and orchestrates the ECG Digitization pipeline stages and delegates tasks to specialized subagents.
---

# Master Orchestrator Skill

This skill governs the planning, stage coordination, and orchestration of the ECG Digitization pipeline execution.

## 📋 Objective
To maintain the end-to-end flow of the ECG digitization pipeline (Stages 1 through 5.2 and future Stages 6 & 7) by orchestrating specialized agents and resolving integration issues.

## ⚙️ Workflow & Integration Steps
1. **Goal Assessment**: Read the pipeline target and assess the active codebase context.
2. **Delegate Planning**: Instruct the **Project Manager** to build a `task.md` roadmap.
3. **Environment Setup**: Instruct the **Branch and Merge Manager** to spin up an isolated git branch for the current task/experiment.
4. **Execution Delegation**: Call the **Experiment Manager** to execute stages and configure MLflow tracking.
5. **Visualization & Validation**:
   - Instruct the **Visualization Manager** to generate visual overlays.
   - Instruct the **Validation Agent** to run verification metrics check.
6. **Documentation & Merging**:
   - Instruct the **Research Manager** to update documentation.
   - Instruct the **Branch and Merge Manager** to clean and merge the branch.
