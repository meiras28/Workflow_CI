import time
import random
from prometheus_client import start_http_server, Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    'api_requests_total', 
    'Total number of API requests'
)
MODEL_PREDICTION_SPEED = Gauge(
    'model_prediction_speed_seconds', 
    'Time taken to predict banking churn'
)
CHURN_PROBABILITY = Histogram(
    'model_churn_probability', 
    'Distribution of predicted churn probabilities'
)

def process_request():
    REQUEST_COUNT.inc()
    
    speed = random.uniform(0.01, 0.25)
    MODEL_PREDICTION_SPEED.set(speed)
    
    prob = random.random()
    CHURN_PROBABILITY.observe(prob)
    
    time.sleep(1)

if __name__ == '__main__':
    start_http_server(8000)
    print("=== SERVER EXPORTER PROMETHEUS AKTIF DI PORT 8000 ===")
    while True:
        process_request()