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

from zenml.pipelines import pipeline
import yaml

@pipeline
def gitflow_training_pipeline(
    get_historical_data,
    get_validation_data,
    model_trainer,
    model_scorer,
    model_register,
    model_deployer,
    deployment_decision
):
    """Pipeline that trains and evaluates a new model."""
    data = get_historical_data()
    model, train_accuracy = model_trainer(train_dataset=data)
    validation_data = get_validation_data()
    test_accuracy = model_scorer(dataset=validation_data, model=model)
    
    model_register(
        model=model
    )
    
    model_deployer(model=model, deploy_decision=deployment_decision(test_accuracy))
    
    print("Accuracy: ", test_accuracy)
