import mlflow
from mlflow.tracking.client import MlflowClient
import os


class MlflowRegistryError(Exception):
    pass


class MlflowRegistry(MlflowClient):
    def __init__(self, registry_uri=None, *args, **kwargs):
        self.client = MlflowClient(registry_uri)
        super().__init__(registry_uri=registry_uri, *args, **kwargs)

    def list_experiments(self, query):
        """Query the mlflow api and return
           a list of experiment dictionaries"""
        runs_list = []
        search_result = self.client.search_runs(experiment_ids="0",
                                                filter_string=query)
        for run in search_result:
            runs_list.append(run.to_dictionary())
        return runs_list

    def register_model(
            self,
            query,
            name="latest-reg-model"):
        """Register most recent model returned from a query"""
        search_result = self.list_experiments(query=query)
        try:
            artifact_uri = search_result[0]['info']['artifact_uri']
        except IndexError:
            raise MlflowRegistryError("Search returned no results."
                                      "Try a different query.")
        mlflow.register_model(artifact_uri,
                              name=name)

    def list_models(self):
        """Return a list of registered models"""
        registered_models = []
        for rm in self.client.list_registered_models():
            registered_models.append(rm)
        return registered_models

    def _find_model(self, name=None, version=None):
        """ Find registered model based on supplied arguments
            Return model's path """
        filter_str = f"name='{name}'"
        if version:
             models = self.client.search_model_versions(filter_string=filter_str)
             if not models:
                 raise MlflowRegistryError(f'Model named {name} does not exist')
             model = list(filter(lambda model: model.version == version,  models))
             try:
                 model_path = os.path.join(model[0].source, 'model')
             except IndexError:
                raise MlflowRegistryError((f'Model named {name},'
                                           f'version {version} does not exist')) from None
        else:
            try:
               models = self.list_models()
               latest = list(filter(lambda model: model.name == name, models))[0]  
            except IndexError:
               raise MlflowRegistryError(f'Model named {name} does not exist') from None
            model_path = os.path.join(latest.latest_versions[0].source, 'model')
        return model_path
	
    def load_model(self, name=None, version=None):
         """ Load registered model based on supplied arguments """
         model_path = self._find_model(name=name, version=version)
         return mlflow.pyfunc.load_model(model_path)
         
