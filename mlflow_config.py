import mlflow
import os

def setup_mlflow(experiment_name="ECG_Digitization"):
    """
    Sets up MLflow tracking. Creates the experiment if it doesn't exist.
    Returns the experiment ID.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
        
    mlflow.set_experiment(experiment_name)
    return experiment_id

def log_dataset_lineage(
    dataset_version="unknown",
    stage4_version="unknown",
    stage5_1_version="unknown",
    stage5_2_version="unknown",
    biomarker_version="unknown",
    calibration_version="unknown",
    patient_split_version="unknown"
):
    """
    Logs strict dataset lineage for the run.
    """
    mlflow.log_params({
        "dataset_version": dataset_version,
        "stage4_version": stage4_version,
        "stage5.1_version": stage5_1_version,
        "stage5.2_extraction_version": stage5_2_version,
        "biomarker_version": biomarker_version,
        "calibration_version": calibration_version,
        "patient_split_version": patient_split_version
    })

def log_standard_metrics(metrics_dict):
    """
    Helper to log standard metrics. 
    Only pass metrics that apply to the model/task.
    """
    # Filter out None values just in case
    valid_metrics = {k: v for k, v in metrics_dict.items() if v is not None}
    if valid_metrics:
        mlflow.log_metrics(valid_metrics)
