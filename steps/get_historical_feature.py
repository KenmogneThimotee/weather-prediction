from datetime import datetime
from typing import Any, Dict, List, Union
import pandas as pd

from zenml.steps import step, BaseParameters
from zenml.pipelines import pipeline
from zenml.client import Client
import yaml


class HistoricalFeatureParams(BaseParameters):
    
    city_name: str
    stage: str

@step()
def get_historical_features(params: HistoricalFeatureParams) -> pd.DataFrame:
    """Feast Feature Store historical data step

    Returns:
        The historical features as a DataFrame.
    """
    feature_store = Client().active_stack.feature_store
    if not feature_store:
        raise ValueError(
            "The Feast feature store component is not available. "
            "Please make sure that the Feast stack component is registered as part of your current active stack."
        )
        
    with open('steps/step_params.yml', 'r') as stream:
        config = yaml.load(stream=stream, Loader=yaml.Loader)
    
    city = pd.read_csv(f"{config['historical_features']['entity_folder']}/{params.city_name}_entity.csv")
    city = city.rename(columns={'datetime': 'event_timestamp'})
    city['event_timestamp']= pd.to_datetime(city['event_timestamp'])
    
    if params.stage == 'trainer':
        city = city[(city['event_timestamp'] >= datetime(config['historical_features']['trainer']['start']['year'],
                                                         config['historical_features']['trainer']['start']['month'],
                                                         config['historical_features']['trainer']['start']['day'])) & (city['event_timestamp'] <= datetime(
                                                             config['historical_features']['trainer']['end']['year'],
                                                             config['historical_features']['trainer']['end']['month'],
                                                             config['historical_features']['trainer']['end']['day']))]
    
    if params.stage == 'validator':
        city = city[(city['event_timestamp'] >= datetime(config['historical_features']['validator']['start']['year'],
                                                         config['historical_features']['validator']['start']['month'],
                                                         config['historical_features']['validator']['start']['day'])) & (city['event_timestamp'] <= datetime(
                                                             config['historical_features']['validator']['end']['year'],
                                                             config['historical_features']['validator']['end']['month'],
                                                             config['historical_features']['validator']['end']['day']))]
    
    if params.stage == 'tester':
        city = city[(city['event_timestamp'] >= datetime(config['historical_features']['tester']['start']['year'],
                                                         config['historical_features']['tester']['start']['month'],
                                                         config['historical_features']['tester']['start']['day'])) & (city['event_timestamp'] <= datetime(
                                                             config['historical_features']['tester']['end']['year'],
                                                             config['historical_features']['tester']['end']['month'],
                                                             config['historical_features']['tester']['end']['day']))]
    
    features = [
        "weather:humidity",
        "weather:temperature",
        "weather:pressure",
        "weather:wind_direction",
        "weather:wind_speed",
    ]
    entity_df = city 

    data =  feature_store.get_historical_features(
        entity_df=entity_df,
        features=features,
        full_feature_names=config['historical_features']['full_feature_names'],
    )
    
    print("Data 2 : ", data)
    
    
    
    
    return data[['humidity', 'temperature', 'pressure', 'wind_direction', 'wind_speed']]


