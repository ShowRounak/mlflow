#### run these commands from CMD
` mkdir mlflow `

` cd mlflow `

` code . `
#### -----------------------------------------------------------------------------------------------
#### bash terminal
` conda create -p env python=3.8 -y `

` source activate ./env `

#### -----------------------------------------------------------------------------------------------
#### Shortcut
` conda create -p env python=3.8 -y && source activate ./env `

#### -----------------------------------------------------------------------------------------------
#### Create requirements.txt file
` touch requirements_mlflow.txt `

#### -----------------------------------------------------------------------------------------------
#### Install requirements.txt file
` pip install -r requirements_mlflow.txt `


#### -----------------------------------------------------------------------------------------------
#### check all python libraries
`pip list `

#### -----------------------------------------------------------------------------------------------
#### To see Mlflow GUI
` mlflow ui `

#### -----------------------------------------------------------------------------------------------
## Some Important mlflow methods
- mlflow.start_run(): Begins a new MLflow run to track an experiment.

- mlflow.log_param(key, value): Logs a single parameter (key-value pair) for the current run.

- mlflow.log_params(params): Logs multiple parameters (dictionary of key-value pairs) for the current run.

- mlflow.log_metric(key, value, step=None): Logs a single metric (key-value pair) for the current run at a specific step.

- mlflow.log_metrics(metrics, step=None): Logs multiple metrics (dictionary of key-value pairs) for the current run at a specific step.

- mlflow.log_artifact(local_path, artifact_path=None): Logs a local file or directory as an artifact associated with the current run.

- mlflow.log_artifacts(local_dir, artifact_path=None): Logs all the files and subdirectories in a local directory as artifacts associated with the current run.

- mlflow.set_tag(key, value): Adds a tag (key-value pair) to the current run for additional metadata.
- mlflow.set_tracking_uri(uri): Sets the URI of the MLflow server for remote tracking and storage.
- mlflow.get_tracking_uri(): Retrieves the URI of the currently configured MLflow server.
- mlflow.set_experiment(experiment_name): Sets the active experiment to the specified name.
- mlflow.start_run(run_id=None, experiment_id=None): Starts a run with a specific ID or in a specific experiment.
- mlflow.end_run(): Ends the active run.
