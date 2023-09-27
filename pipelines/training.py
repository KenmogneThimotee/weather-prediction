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

from zenml import pipeline
import yaml
from steps.get_historical_feature import get_historical_features
from steps.model_trainers import (
    decision_tree_trainer,
)

from steps.model_evaluators import (
    model_scorer,
    deployment_decision
)

from zenml.integrations.mlflow.steps.mlflow_registry import (
        mlflow_register_model_step
    )
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step 
from utils.tracker_helper import get_current_tracker_run_url
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

def load_config():
    
    with open('steps/step_params.yml', 'r') as stream:
        config = yaml.load(stream=stream, Loader=yaml.Loader)
        
    return config['city']


@pipeline(enable_cache=False)
def gitflow_training_pipeline(city: str):
    """Pipeline that trains and evaluates a new model."""
    
    data = get_historical_features(city_name=city, stage="trainer")
    model = decision_tree_trainer(train_dataset=data, model_city=city)
    validation_data = get_historical_features(city_name=city, stage="validator")
    test_accuracy = model_scorer(accuracy_metric_name="test_accuracy", dataset=validation_data, model=model)
    

    mlflow_register_model_step(
        model=model,
        name=f"{city}-model",
        trained_model_name=f"{city}-model",
        experiment_name="default"
        )
    
    # mlflow_model_deployer_step(model=model, deploy_decision=deployment_decision(test_accuracy))
    
    print("Accuracy: ", test_accuracy)


def concurent_pipeline(city: str):
    
    
    @pipeline(name=city, enable_cache=False, enable_artifact_metadata=True)
    def gitflow_training_pipeline():
        """Pipeline that trains and evaluates a new model."""
        
        data = get_historical_features(city_name=city, stage="trainer")
        model = decision_tree_trainer(train_dataset=data, model_city=city)
        validation_data = get_historical_features(city_name=city, stage="validator")
        test_accuracy = model_scorer(accuracy_metric_name="test_accuracy", dataset=validation_data, model=model)
        
        mlflow_register_model_step(
            model=model,
            name=f"{city}-model",
            trained_model_name="model",
            experiment_name=city
        )
        
        mlflow_model_deployer_step(model=model, deploy_decision=deployment_decision(test_accuracy))
        
        print("Accuracy: ", test_accuracy)
    
    gitflow_training_pipeline.prepare()
    gitflow_training_pipeline()

