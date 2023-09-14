#!/usr/bin/env bash

set -Eeo pipefail

zenml data-validator register deepchecks_data_validator --flavor=deepchecks
zenml experiment-tracker register local_mlflow_tracker  --flavor=mlflow
zenml model-deployer register local_mlflow_deployer  --flavor=mlflow
zenml feature-store register feast_store --feast_repo="feature_store_repo/feature_repo"  --flavor=feast
zenml stack register local_gitflow_stack \
    -a default \
    -o default \
    -e local_mlflow_tracker \
    -d local_mlflow_deployer \
    -dv deepchecks_data_validator \
    -f feast_store
zenml stack set local_gitflow_stack