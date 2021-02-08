from unittest.mock import patch
from mlflow.entities import (RunData,
                             RunInfo,
                             Run,
                             RunStatus,
                             Metric,
                             Param,
                             RunTag)

from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from mlflow.pyfunc import PyFuncModel
import pytest

from corescore.mlflowregistry import MlflowRegistry, MlflowRegistryError


def mock_search_registered_models(*args, **kwargs):
    response = [RegisteredModel(name='corescore',
                                latest_versions=[ModelVersion(name='corescore',
                                version='1',
                                creation_timestamp=222,
                                current_stage='Production',
                                source='/var/lib')])]*2
    return response

def mock_search_model_versions(*args, **kwargs):
    if 'corescore' in kwargs['filter_string']:
        response = [ModelVersion(name='corescore',
                                 version='1',
                                 creation_timestamp=222,
                                 current_stage='Production',
                                 source='/var/lib/')]*2
        return response
    else:
        return []


def mock_response(*args, **kwargs):
    if 'corescore' in kwargs['filter_string']:
        mock_run = Run(
            run_data=RunData(
                metrics=[
                    Metric(
                        key='value_loss',
                        value=2.2,
                        timestamp=5000,
                        step=31),
                    Metric(
                        key='acc',
                        value=42,
                        timestamp=2131,
                        step=3)],
                params={
                    Param(
                        key='batch_size',
                        value='2')},
                tags=[
                    RunTag(
                        key='model',
                        value='corescore')]),
            run_info=RunInfo(
                run_uuid='run_uuid',
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


@patch('mlflow.tracking.client.MlflowClient.list_registered_models',
       return_value=[RegisteredModel('test_model')])
def test_list_models(*args):
    client = MlflowRegistry()
    assert isinstance(client.list_models(), list)
    assert isinstance(client.list_models()[0], RegisteredModel)


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


@patch('mlflow.tracking.client.MlflowClient.search_model_versions',
       side_effect =mock_search_model_versions)
@patch('mlflow.tracking.client.MlflowClient.list_registered_models',
       side_effect=mock_search_registered_models)
def test_find_model_error(*args):
    client = MlflowRegistry()
    with pytest.raises(MlflowRegistryError) as exc_info:
         model_path = client._find_model(name='blah')
    assert 'named' in str(exc_info.value)
    
    with pytest.raises(MlflowRegistryError) as exc_info:
         model_path = client._find_model(name='corescore', version='9999')
    assert 'version' in str(exc_info.value)
    
    with pytest.raises(MlflowRegistryError) as exc_info:
         model_path = client._find_model(name='blah', version='1')
    assert 'blah' in str(exc_info.value)    


@patch('mlflow.tracking.client.MlflowClient.list_registered_models',
       side_effect=mock_search_registered_models)
@patch('mlflow.tracking.client.MlflowClient.search_model_versions',
       side_effect=mock_search_model_versions)
def test_find_model(*args):
    client = MlflowRegistry()
    model_path = client._find_model(name='corescore')
    assert '/var/' in model_path
    model_path = client._find_model(name='corescore', version='1')
    assert '/var/' in model_path


@patch('mlflow.pyfunc.load_model', return_value=PyFuncModel)
@patch('corescore.mlflowregistry.MlflowRegistry._find_model', return_value='/var/lib/mlflow/model')
def test_load_model(*args):
    client = MlflowRegistry()
    model = client.load_model()
    assert model == PyFuncModel 

