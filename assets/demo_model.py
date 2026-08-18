# Placeholder for model loading logic
class DemoModel:
    def __init__(self):
        self.layers = [
            {"name": "Input", "params": 784},
            {"name": "Dense_1", "params": 128},
            {"name": "BatchNorm", "params": 256},
            {"name": "Dropout", "params": "0.2"},
            {"name": "Output", "params": 10}
        ]