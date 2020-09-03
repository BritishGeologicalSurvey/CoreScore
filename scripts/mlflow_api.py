#Use mlflow api to get best metric
import os
from mlflow.tracking.client import MlflowClient


def best_metric(client, metric):
    search_result = client.search_runs(experiment_ids="0", max_results=1, order_by=[f"metrics.{metric} DESC"])[0]
    try: 
        metric_best = search_result.to_dictionary()['data']['metrics'][f'{metric}']
        return metric_best
    except KeyError:  
        raise KeyError("Metric not found, please provide a different metric name")


if __name__ == "__main__":
   URI = os.environ.get('MLFLOW_TRACKING_URI', '') 
   client = MlflowClient(URI)
   print(best_metric(client, 'acc_rock'))



