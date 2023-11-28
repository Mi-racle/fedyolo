import os
import sys
from collections import OrderedDict

import flwr as fl
import torch
from flwr.server.strategy import Strategy

from ultralytics import YOLO

cur_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur_dir, '..'))
sys.path.append(root)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_YAML = 'D:/xxs-signs/fedyolo/datasets/raw_images/data.yaml'


class CifarClient(fl.client.NumPyClient):

    def __init__(self):
        super().__init__()
        self.net = YOLO('yolov8n.yaml')
        self.map = 1

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self.train(epochs=1, device=DEVICE)
        return self.get_parameters(config={}), self.map, metrics.results_dict

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self.test(device=DEVICE)
        self.map = metrics.results_dict['metrics/mAP50-95(B)']
        return 1 - metrics.results_dict['metrics/mAP50-95(B)'], self.map, metrics.results_dict

    def train(self, epochs, device):
        """Train the network on the training set."""
        return self.net.train(data=DATA_YAML, epochs=epochs, imgsz=640, workers=0, device=device)

    def test(self, device):
        """Validate the network on the entire test set."""
        return self.net.val(data=DATA_YAML, device=device)


fl.client.start_numpy_client(server_address="localhost:8080", client=CifarClient())
