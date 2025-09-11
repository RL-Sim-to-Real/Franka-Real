import pandas as pd

class MetricLogger:
    def __init__(self):
        self.metrics = []

    def clear(self):
        self.metrics = []

    def log(self, metric_dict):
        self.metrics.append(metric_dict)

    def pop(self, count):
        self.metrics = self.metrics[:-count]

    def save(self, filename):
        df = pd.DataFrame(self.metrics)
        df.to_csv(filename, index=False)
