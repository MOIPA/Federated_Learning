import numpy as np
from flask import Flask, request, jsonify
from Model import Net18
import torch
import torch.nn as nn
import pickle
import base64
import requests
import time
import multiprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from queue import Queue
from aggregation_solution import weighted_average_aggregation, average_aggregation

from tools import plot_accuracy_vs_epoch
from matplotlib import pyplot as plt


num_output_features = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


a = Net18(num_output_features).to(device)
a.load_state_dict(torch.load("test_aggregated_model.pt"))

b = Net18(num_output_features).to(device)
b.load_state_dict(torch.load("aggregated_global_model.pt"))
# Get the state_dict of both models
state_dict_a = a.state_dict()
state_dict_b = b.state_dict()

# Assume the models are the same until proven otherwise
models_are_same = True

# Compare the state_dict of both models
for key in state_dict_a:
    if key in state_dict_b:
        # If the key is in both state_dict, compare the tensors
        if not torch.equal(state_dict_a[key], state_dict_b[key]):
            print(f"Difference detected in {key}")
            models_are_same = False
    else:
        # If the key is not in state_dict_b, the models are not the same
        print(f"{key} is missing in the second model")
        models_are_same = False

if models_are_same:
    print("The models are the same")
else:
    print("The models are not the same")
