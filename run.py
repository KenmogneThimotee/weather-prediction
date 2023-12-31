from zenml.client import Client
from zenml.config import DockerSettings

from pipelines import (
    gitflow_training_pipeline,
    concurent_pipeline
)

from zenml.enums import ExecutionStatus
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.deepchecks import DeepchecksIntegration
from utils.tracker_helper import LOCAL_MLFLOW_UI_PORT, get_tracker_name
import yaml
from yaml import Loader

from zenml.integrations.mlflow.steps.mlflow_registry import (
        mlflow_register_model_step
    )
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step 

# These global parameters should be the same across all workflow stages.
RANDOM_STATE = 23
TRAIN_TEST_SPLIT = 0.2
MIN_TRAIN_ACCURACY = 0.9
MIN_TEST_ACCURACY = 0.9
MAX_SERVE_TRAIN_ACCURACY_DIFF = 0.1
MAX_SERVE_TEST_ACCURACY_DIFF = 0.05
WARNINGS_AS_ERRORS = False

def load_config():
    
    with open('steps/step_params.yml', 'r') as stream:
        config = yaml.load(stream=stream, Loader=Loader)
        
    return config['city']

from multiprocessing import Pool
from threading import Thread
def main(
    disable_caching: bool = False,
    requirements_file: str = "requirements.txt",
):  
    
    with Pool(5) as p:
        for city in load_config():
            p.apply(concurent_pipeline, args=[city])
        
    # with Pool(5) as p:
    #      print(p.map(concurent_pipeline, load_config()))

if __name__ == "__main__":
    main()
