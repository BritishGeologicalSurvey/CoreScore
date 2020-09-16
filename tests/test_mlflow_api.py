from unittest.mock import patch
from mlflow.entities import (RunData,
                             RunInfo,
                             Run,
                             RunStatus,
                             Metric,
                             Param,
                             RunTag)
from mlflow.entities.model_registry import RegisteredModel
import pytest

from corescore.mlflowregistry import MlflowRegistry, MlflowRegistryError


@patch('mlflow.tracking.client.MlflowClient.list_registered_models',
       return_value=[RegisteredModel('test_model')])
def test_list_models(*args):
    client = MlflowRegistry()
    assert isinstance(client.list_models(), list)
    assert isinstance(client.list_models()[0], RegisteredModel)


def mock_response(*args, **kwargs):
    if "corescore" in kwargs['filter_string']: 
        mock_run = Run(run_data=RunData(metrics=[Metric(key='value_loss',
                                                    value=2.2,
                                                    timestamp=5000,
                                                    step=31),
                                             Metric(key='acc',
                                                    value=42,
                                                    timestamp=2131,
                                                    step=3)],
                                    params={Param(key='batch_size',
                                                  value='2')},
                                    tags=[RunTag(key='model', value='corescore')]),
                   run_info=RunInfo(run_uuid='run_uuid',
                                    experiment_id='experiment_id',
                                    user_id='user_id',
                                    status=RunStatus(),
                                    start_time=2222,
                                    end_time=2222,
                                    artifact_uri='artifact_uri',
                                    lifecycle_stage='testing'))
        return [mock_run] 
    else:
        return []     


@patch('mlflow.tracking.client.MlflowClient.search_runs',
       side_effect=mock_response)
def test_list_experiments(*args):
    client = MlflowRegistry()
    response = client.list_experiments(query='tag.model = "corescore"')
    assert isinstance(response, list)
    assert isinstance(response[0], dict)


@patch('mlflow.tracking.client.MlflowClient.search_runs',
       side_effect=mock_response)
def test_register_model(*args):
    client = MlflowRegistry()
    with pytest.raises(MlflowRegistryError):
        assert client.register_model(query='tag.model = "random_name"')
