import mlflow
from mlflow.tracking.client import MlflowClient
import os


class MlflowRegistryError(Exception):
    pass


class MlflowRegistry(MlflowClient):
    def __init__(self, client=None, *args, **kwargs):
        self.client = MlflowClient(client)
        super().__init__(*args, **kwargs)

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
            metric,
            override=False,
            name="latest-reg-model"):
        """Register a model only if it shows better performance"""
        search_result = self.list_experiments(query=query)
        old_metric = search_result[1]['data']['metrics'][f"{metric}"]
        latest_metric = search_result[0]['data']['metrics'][f"{metric}"]
        if old_metric >= latest_metric and not override:
            msg = (f"Model did not show improvement."
                   f"Model was not registered."
                   f"Set `override=True` to override.")
            print(msg)
        else:
            mlflow.register_model(search_result[0]['info']['artifact_uri'],
                                  name)

    def list_models(self):
        """Return a list of registered models"""
        registered_models = []
        for rm in self.client.list_registered_models():
            registered_models.append(rm)
        return registered_models

    def load_model(self, name):
        """Load model from registry based on name"""
        models = self.list_models()
        latest = list(filter(lambda model: model.name == name, models))[0]
        model_path = os.path.join(latest.latest_versions[0].source, 'model')
        print(model_path)
        model = mlflow.fastai.load_model(model_path)
