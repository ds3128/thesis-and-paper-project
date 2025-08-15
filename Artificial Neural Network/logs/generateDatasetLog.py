import faker
from datetime import datetime, timedelta
import random
from constant import Constant


def generate_datetimes(start_date: datetime, end_date: datetime, nb_logs: int) -> list[datetime]:
    timestamps = []
    total_seconds = (end_date - start_date).total_seconds()
    for _ in range(nb_logs):
        random_offset = random.random() * total_seconds
        timestamp = start_date + timedelta(seconds=random_offset)
        timestamps.append(timestamp)
    timestamps.sort()
    return timestamps


def generate_anomaly_interval(
    number_of_anomaly_intervals: int,
    number_of_logs: int,
    min_number_of_anomaly_per_intervals: int,
    max_number_of_anomaly_per_intervals: int,
    anomaly_types: list[str]
) -> list[dict[str, int | str]]:
    
    intervals = []
    
    for _ in range(number_of_anomaly_intervals):
        start_idx = random.randint(0, number_of_logs - max_number_of_anomaly_per_intervals)
        nb_anomaly = random.randint(min_number_of_anomaly_per_intervals, max_number_of_anomaly_per_intervals)
        end_idx = min(start_idx + nb_anomaly, number_of_logs - 1)
        
        intervals.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "type": random.choice(anomaly_types)
        })
    return intervals


def generate_logs_datasets(
    start_date: datetime, 
    end_date: datetime,
    number_of_anomaly_intervals: int,
    number_of_logs: int,
    min_number_of_anomaly_per_intervals: int,
    max_number_of_anomaly_per_intervals: int,
    http_error_list: list[str],
    number_of_anomaly_ips: int,
    logs_dataset_filename: str,
    http_methods: list[str],
    http_normal_codes: list[str],
    http_error_code: dict[str, list[str]],
    api_end_points: list[str]
):
    fake = faker.Faker()
    
    anomaly_ips = [fake.ipv4() for _ in range(number_of_anomaly_ips)]
    
    timestamps = generate_datetimes(
        start_date=start_date,
        end_date=end_date, 
        nb_logs=number_of_logs  # Correction ici
    )
    
    anomaly_types = http_error_list.copy()
    anomaly_types.append("mixed_errors")
    
    anomaly_intervals = generate_anomaly_interval(
        number_of_anomaly_intervals=number_of_anomaly_intervals,
        number_of_logs=number_of_logs,  # Correction ici
        min_number_of_anomaly_per_intervals=min_number_of_anomaly_per_intervals,
        max_number_of_anomaly_per_intervals=max_number_of_anomaly_per_intervals,
        anomaly_types=anomaly_types
    )
    
    with open(logs_dataset_filename, "w") as file:
        file.write("timestamps,user_ip,method,status_code,end_point,response_time\n")
        
        for i in range(number_of_logs):
            timestamp = timestamps[i]
            user_ip = random.choice(anomaly_ips)  # Correction ici
            method = random.choice(http_methods)
            status_code = random.choice(http_normal_codes)
            end_point = random.choice(api_end_points)
            response_time = random.randint(10, 300)
            
            for interval in anomaly_intervals:
                if interval["start_idx"] <= i <= interval["end_idx"]:
                    if interval["type"] == "server_errors":
                        status_code = random.choice(http_error_code["server_errors"])
                    elif interval["type"] == "client_errors":
                        status_code = random.choice(http_error_code["client_errors"])
                    elif interval["type"] == "timeout_errors":
                        status_code = random.choice(http_error_code["timeout_errors"])
                        response_time = random.randint(1000, 5000)
            file.write(f"{timestamp},{user_ip},{method},{status_code},{end_point},{response_time}\n")
    
    print(f"[OK] le fichier CSV '{logs_dataset_filename}' a été généré avec {number_of_logs} logs et {number_of_anomaly_intervals} plages d'anomalies.")


if __name__ == "__main__":
    generate_logs_datasets(
        start_date=Constant.LOGS_START_DATE,
        end_date=Constant.LOGS_END_DATE,
        number_of_anomaly_intervals=Constant.NUMBER_OF_ANOMALY_INTERVALS,
        number_of_logs=Constant.NUMBER_LOGS,
        min_number_of_anomaly_per_intervals=Constant.MIN_NUMBER_OF_ANOMALY_PER_INTERVAL,
        max_number_of_anomaly_per_intervals=Constant.MAX_NUMBER_OF_ANOMALY_PER_INTERVAL,
        http_error_list=list(Constant.HTTP_ERRORS_CODES.keys()),
        number_of_anomaly_ips=Constant.NUMBER_OF_ANOMALY_IPS,
        logs_dataset_filename=Constant.LOGS_DATASET_FILENAME,
        http_methods=Constant.HTTP_METHODS,
        http_normal_codes=Constant.HTTP_NORMAL_CODES,
        http_error_code=Constant.HTTP_ERRORS_CODES,
        api_end_points=Constant.API_ENDPOINTS
    )
