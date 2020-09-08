#Use mlflow api to get best metric
import os
from mlflow.tracking.client import MlflowClient
import mlflow

class MlflowRegistryError(Exception):
    pass


def get_best(client, tag, search_str, metric):
    query = f"tags.{tag} = '{search_str}'"
    search_result = client.search_runs(experiment_ids="0", filter_string=query, max_results=1, order_by=[f"metrics.{metric} DESC"])[0]
    artifact_uri = search_result.to_dictionary()['info']['artifact_uri']
    run_id = search_result.to_dictionary()['info']['run_id']
    return {'artifact_uri': artifact_uri, 'run_id': run_id}


def get_latest(client, tag, search_str):
    query = f"tags.{tag} = '{search_str}'" 
    search_res = client.search_runs(experiment_ids="0", filter_string=query)[0]
    return search_res

def register_best(client, tag, search_str, metric, name="best-reg-model"):
    best_model = get_best(client=client,
                          tag=tag,
			  search_str=search_str,
                          metric=metric)
    res = mlflow.register_model(best_model['artifact_uri'],
                                name)
    return res


def register_latest(client, tag, search_str, name="latest-reg-model"):
    latest_model = get_latest(client=client,
                              tag=tag,
                              search_str=search_str)
    mlflow.register_model(latest_model['artifact_uri'],
                          name)


if __name__ == "__main__":
   URI = os.environ.get('MLFLOW_TRACKING_URI', '') 
   client = MlflowClient(URI)
   register_best(client, "model", "corescore", "acc_rock")


