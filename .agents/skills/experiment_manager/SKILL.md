---
name: experiment_manager
description: Tracks runs, configures MLflow parameters, logs model artifacts, and synchronizes notebooks and scripts.
---

# Experiment Manager Skill

This skill governs the execution of experiments, tracking with MLflow, and keeping notebooks/python scripts in sync.

## 📋 Objective
To ensure experimental runs are completely reproducible, parameters are logged properly, and changes in notebooks are ported back to source python scripts (`run_stage*.py`, `train_stage5.py`, etc.).

## ⚙️ Workflow & Steps
1. **Sync Check**: Verify that any development notebooks (`*.ipynb`) are in sync with their production script counterpart (`*.py`).
2. **MLflow Check**: Configure the MLflow tracking URI (default: `file:./mlruns`) and select the `ECG_Digitization` experiment.
3. **Parameter Registration**: Log all input parameters (learning rate, batch size, epoch count, preprocessing filters) to MLflow.
4. **Execution Run**: Trigger the python stage execution (e.g. `run_stage5.py` or `train_stage5.py`).
5. **Artifact Storage**: Ensure that the resulting model files (e.g., `stage5_best_model.pth`) and history files (e.g. `stage5_training_history.json`) are logged as MLflow run artifacts.
