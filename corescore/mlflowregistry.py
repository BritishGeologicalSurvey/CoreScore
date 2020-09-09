import mlflow
from mlflow.tracking.client import MlflowClient

class MlflowRegistryError(Exception):
    pass


class MlflowRegistry(MlflowClient):
    def  __init__(self, client=None, *args,  **kwargs):
        self.client = MlflowClient(client)
        super().__init__(*args, **kwargs)


    def get_info(self, search_result):
        """Given a search response from the mlflow api
           return artifact_uri, run_id and metrics"""
        search_dict = search_result.to_dictionary()
        artifact_uri = search_dict['info']['artifact_uri']
        run_id = search_dict['info']['run_id']
        metrics = search_dict['data']['metrics']
        return {'artifact_uri': artifact_uri, 'run_id': run_id, 'metrics': metrics}

    def list_experiments(self, tag, search_str):
        """Query the mlflow api and list
           logged models"""
        query = f"tags.{tag} = '{search_str}'" 
        return self.client.search_runs(experiment_ids="0",
                                       filter_string=query)
 
    def register_model(self, tag, search_str, metric, override=False, name="latest-reg-model"):
        """Register a model if it shows better performance"""
        exp_list = self.list_experiments(tag=tag,
                                       search_str=search_str)
        old_metric = self.get_info(exp_list[1])['metrics'][f"{metric}"]
        latest_metric = self.get_info(exp_list[0])['metrics'][f"{metric}"]
        if old_metric >= latest_metric and not override:
            print(f"Latest model's {metric} is {latest_metric} and not better than {old_metric}. Set `override=True` to override. Model was not registered.")
        else:
            mlflow.register_model(self.get_info(exp_list[0])['artifact_uri'],
                              name)
