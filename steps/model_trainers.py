#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

"""Model training steps used to train a model on the training data."""

import pandas as pd
from sklearn.base import  RegressorMixin
from sklearn.svm import SVR
from zenml.client import Client
from zenml.steps import BaseParameters, Output, step
from zenml import step

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from steps.data_loaders import DATASET_TARGET_COLUMN_NAME
from utils.tracker_helper import enable_autolog, get_tracker_name
from mlflow.models import infer_signature
import mlflow
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.experiment_trackers import mlflow_experiment_tracker


@step(
    experiment_tracker=get_tracker_name(),
)
def svc_trainer(
    train_dataset: pd.DataFrame,
    random_state: int = 42,
    C: float = 1.320498,
    kernel: str = "rbf",
    degree: int = 3,
    coef0: float = 0.0,
    shrinking: bool = True,
    extra_hyperparams: dict = {},
):
    """Train and logs a sklearn C-support vector classification model.
    
    If the experiment tracker is enabled, the model and the training accuracy
    will be logged to the experiment tracker.

    Args:
        params: The hyperparameters for the model.
        train_dataset: The training dataset to train the model on.

    Returns:
        The trained model and the training accuracy.
    """
    enable_autolog()

    X = train_dataset.drop(columns=[DATASET_TARGET_COLUMN_NAME])
    y = train_dataset[DATASET_TARGET_COLUMN_NAME]
    model = SVR(
        C=C,
        kernel=kernel,
        degree=degree,
        coef0=coef0,
        shrinking=shrinking,
        random_state=random_state,
        **extra_hyperparams,
    )

    model.fit(X, y)
    train_acc = model.score(X, y)
    
    print(f"Train accuracy: {train_acc}")
    return model



@step(
    experiment_tracker=get_tracker_name()
)
def decision_tree_trainer(
    train_dataset: pd.DataFrame,
    model_city: str,
    random_state: int = 42,
    max_depth: int = 5,
    extra_hyperparams: dict = {},
):
    """Train a sklearn decision tree classifier.

    If the experiment tracker is enabled, the model and the training accuracy
    will be logged to the experiment tracker.

    Args:
        params: The hyperparameters for the model.
        train_dataset: The training dataset to train the model on.
    
    Returns:
        The trained model and the training accuracy.
    """
    
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.autolog()

    X = train_dataset.drop(columns=[DATASET_TARGET_COLUMN_NAME])
    y = train_dataset[DATASET_TARGET_COLUMN_NAME]
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state,
        **extra_hyperparams,
    )

    model.fit(X, y)
    train_acc = model.score(X, y)
    print(f"Train accuracy: {train_acc}")
    signature = infer_signature(X, model.predict(X))

    # log model
    mlflow.sklearn.log_model(model, f"{model_city}-model", signature=signature)
    return model
