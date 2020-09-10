import mlflow
from mlflow.tracking.client import MlflowClient
import os


class MlflowRegistryError(Exception):
    pass


class MlflowRegistry(MlflowClient):
    def __init__(self, client=None, *args, **kwargs):
        self.client = MlflowClient(client)
        super().__init__(*args, **kwargs)

    def _get_info(self, search_result):
        """Given a search response from the mlflow api
           return artifact_uri, run_id and metrics"""
        search_dict = search_result.to_dictionary()
        artifact_uri = search_dict['info']['artifact_uri']
        run_id = search_dict['info']['run_id']
        metrics = search_dict['data']['metrics']
        return {
            'artifact_uri': artifact_uri,
            'run_id': run_id,
            'metrics': metrics}

    def list_experiments(self, tag, search_str):
        """Query the mlflow api and list
           logged experiments"""
        query = f"tags.{tag} = '{search_str}'"
        return self.client.search_runs(experiment_ids="0",
                                       filter_string=query)

    def register_model(
            self,
            tag,
            search_str,
            metric,
            override=False,
            name="latest-reg-model"):
        """Register a model if it shows better performance"""
        exp_list = self.list_experiments(tag=tag,
                                         search_str=search_str)
        old_metric = self.get_info(exp_list[1])['metrics'][f"{metric}"]
        latest_metric = self.get_info(exp_list[0])['metrics'][f"{metric}"]
        if old_metric >= latest_metric and not override:
            msg = (f"Latest model's {metric} is {latest_metric}"
                   f"and not better than {old_metric}."
                   f"Set `override=True` to override."
                   f"Model was not registered.")
            print(msg)
        else:
            mlflow.register_model(self.get_info(exp_list[0])['artifact_uri'],
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
