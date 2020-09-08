import mlflow
from mlflow.tracking.client import MlflowClient

class MlflowRegistryError(Exception):
    pass


class MlflowRegistry(MlflowClient):
    def  __init__(self, client=None, *args,  **kwargs):
        self.client = MlflowClient(client)
        super().__init__(*args, **kwargs)

    def get_best(self, client, tag, search_str, metric):
        query = f"tags.{tag} = '{search_str}'"
        search_result = self.client.search_runs(experiment_ids="0", filter_string=query, max_results=1, order_by=[f"metrics.{metric} DESC"])[0]
        artifact_uri = search_result.to_dictionary()['info']['artifact_uri']
        run_id = search_result.to_dictionary()['info']['run_id']
        return {'artifact_uri': artifact_uri, 'run_id': run_id}

    def get_latest(self, client, tag, search_str):
        query = f"tags.{tag} = '{search_str}'" 
        search_res = self.client.search_runs(experiment_ids="0", filter_string=query)[0]
        return search_res

    def register_best(self, client, tag, search_str, metric, name="best-reg-model"):
        best_model = self.get_best(client=client,
                                   tag=tag,
		        	   search_str=search_str,
                                   metric=metric)
        mlflow.register_model(best_model['artifact_uri'],
                              name)


    def register_latest(self, client, tag, search_str, name="latest-reg-model"):
        latest_model = self.get_latest(client=client,
                                       tag=tag,
                                       search_str=search_str)
        mlflow.register_model(latest_model['artifact_uri'],
                              name)
