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
           return artifact_uri and run_id"""
        search_dict = search_result.to_dictionary()
        artifact_uri = search_dict['info']['artifact_uri']
        run_id = search_dict['info']['run_id']
        return {'artifact_uri': artifact_uri, 'run_id': run_id}

    def get_best(self, tag, search_str, metric):
        """Query the mlflow api and return
            best model"""
        query = f"tags.{tag} = '{search_str}'"
        search_result = self.client.search_runs(experiment_ids="0",
                                                filter_string=query,
                                                order_by=[f"metrics.{metric} DESC"])[0]
        return self.get_info(search_result)

    def get_latest(self, tag, search_str):
        """Query the mlflow api and return
           latest model"""
        query = f"tags.{tag} = '{search_str}'" 
        search_result = self.client.search_runs(experiment_ids="0",
                                                filter_string=query)[0]
        return self.get_info(search_result)

    def register_best(self, tag, search_str, metric, name="best-reg-model"):
        """Register best model"""
        best_model = self.get_best(tag=tag,
		        	   search_str=search_str,
                                   metric=metric)
        mlflow.register_model(best_model['artifact_uri'],
                              name)


    def register_latest(self, tag, search_str, name="latest-reg-model"):
        """Register latest model"""
        latest_model = self.get_latest(tag=tag,
                                       search_str=search_str)
        mlflow.register_model(latest_model['artifact_uri'],
                              name)
