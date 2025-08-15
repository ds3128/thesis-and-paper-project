import requests
import pandas as pd

# URL de Prometheus
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

# Requête PromQL pour récupérer toutes les données sur `http_server_requests_seconds_bucket`
PROMQL_QUERY = 'http_server_requests_seconds_bucket'

def fetch_prometheus_data():
    """Interroge Prometheus et récupère les données associées à http_server_requests_seconds_bucket."""
    
    response = requests.get(PROMETHEUS_URL, params={"query": PROMQL_QUERY})
    response_json = response.json()
    
    if "data" not in response_json or "result" not in response_json["data"]:
        print("❌ Erreur : La réponse de Prometheus ne contient pas 'data' ou 'result'.")
        return pd.DataFrame()  # Retourne un DataFrame vide en cas d'erreur

    result = response_json["data"]["result"]
    
    # Liste pour stocker les données
    data = []

    for item in result:
        if isinstance(item, dict) and "metric" in item and "value" in item:
            metric = item["metric"]
            value = item["value"]

            # Extraction du timestamp et de la valeur
            timestamp = value[0]  # Premier élément : timestamp
            metric_value = value[1]  # Deuxième élément : valeur

            # Création d'un dictionnaire contenant toutes les informations
            row = {
                "timestamp": timestamp,
                "value": metric_value,
                **metric  # Déstructure tous les labels
            }

            data.append(row)

    # Création du DataFrame
    df = pd.DataFrame(data)

    # Convertir le timestamp en format lisible
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    return df

# Récupérer les métriques
df_metrics = fetch_prometheus_data()

# Afficher les premières lignes
print(df_metrics.head())

# Sauvegarde en CSV
df_metrics.to_csv("prometheus_metrics.csv", index=False)


# import requests
# import pandas as pd
# import time

# # URL de l'API Prometheus
# PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

# # Requête pour récupérer les métriques
# QUERIES = {
#     "response_time": 'http_server_requests_seconds_sum',
#     "request_count": 'http_server_requests_seconds_count'
# }

# def fetch_prometheus_data(query):
#     """Récupère les données de Prometheus pour une métrique donnée."""
#     response = requests.get(PROMETHEUS_URL, params={"query": query})
#     data = response.json()
    
#     # Vérifier si la réponse contient des résultats
#     if "data" in data and "result" in data["data"]:
#         return data["data"]["result"]
#     return []

# def process_data():
#     """Transforme les données récupérées en un DataFrame."""
#     response_times = fetch_prometheus_data(QUERIES["response_time"])
#     request_counts = fetch_prometheus_data(QUERIES["request_count"])

#     data_list = []

#     for rt, rc in zip(response_times, request_counts):
#         metric = rt["metric"]
#         instance = metric.get("instance", "unknown")
#         method = metric.get("method", "unknown")
#         status = metric.get("status", "unknown")
#         uri = metric.get("uri", "unknown")
        
#         time_value = float(rt["value"][1])  # Temps total
#         count_value = float(rc["value"][1])  # Nombre de requêtes
        
#         response_time = time_value / count_value if count_value > 0 else 0
        
#         data_list.append([time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), instance, method, status, uri, response_time])

#     # Création du DataFrame
#     df = pd.DataFrame(data_list, columns=["timestamp", "instance", "method", "status", "uri", "response_time"])

#     return df

# # Sauvegarde en CSV
# df_metrics = process_data()
# df_metrics.to_csv("prometheus_metrics.csv", index=False)

# print("Données collectées et enregistrées dans 'prometheus_metrics.csv'.")

