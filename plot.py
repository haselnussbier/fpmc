import csv
from datetime import datetime
TIMESTAMP: str

def instantiate_training_csv():
    now = datetime.now()
    global TIMESTAMP
    TIMESTAMP = '{:%H:%M}'.format(now)
    with open("results/training-"+TIMESTAMP+".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss', 'util', 'p'])


def write_data(data):
    global TIMESTAMP
    with open("results/training-" + TIMESTAMP + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data)

