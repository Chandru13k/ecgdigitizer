---
name: branch_merge_manager
description: Manages Git branches, pulls, commits, merges, and ensures master/main branches remain clean.
---

# Branch & Merge Manager Skill

This skill governs Git repository lifecycle management.

## 📋 Objective
To enforce clean branching rules, prevent broken code from landing on the main branch, and manage experiment branch isolation.

## ⚙️ Workflow & Steps
1. **Branch Creation**: Create experiment branches using standard naming rules (e.g. `experiment/stage<num>_<description>`).
2. **Exclude Check**: Verify that gitignore patterns correctly prevent binary weights, dynamic dataset runs, and temporary logs from being committed.
3. **Audit Gate**: Before merging, query the **Validation Agent** to confirm all test passes.
4. **Merge Execution**: Perform clean merges, resolving conflicts carefully. Delete stale experiment branches once merged.
