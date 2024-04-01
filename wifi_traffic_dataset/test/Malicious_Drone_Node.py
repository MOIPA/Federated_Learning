import requests
import torch
import torch.nn as nn
import torch.optim as optim
from Model import Net18
import pickle
import io
import base64
from flask import Flask, request, jsonify
from multiprocessing import *
import numpy as np
import sys
import time
from Model import Net18
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from Dataset import n_nodes
from tools import plot_accuracy_vs_epoch
import copy



torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_output_features = 2

# 对于每个节点，加载训练和测试数据


class MaliciousDroneNode:
    def __init__(self, drone_id):
        self.port = 5001
        self.central_server_ip = "localhost:5000"
        self.drone_id = 2
        self.local_data = None
        self.local_model = None
        self.performance = None
        # flask启动的时候data obejct会丢失，因此再服务器运行的时候定义。

    """DroneNode类在接收到全局模型时,将使用global_model的权重克隆一份副本,并将其分配给self.local_model.
这样，每个无人机节点都可以在本地训练自己的模型副本，并在训练完成后将其上传给中心服务器。
中心服务器可以聚合这些本地模型，从而更新全局模型."""

    def receive_global_model(self, global_model):
        self.global_model = global_model
        self.local_model = Net18(num_output_features).to(device)
        self.local_model.load_state_dict(global_model.state_dict())

    def train_local_model(self):
        if self.local_model is None:
            print("Error: No local model is available for training.")
            return

        # 定义损失函数和优化器

        num_epochs = 15
        learning_rate = 0.01
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=learning_rate)

        if self.local_model.num_output_features == 1:
            criterion = nn.BCEWithLogitsLoss()
        elif self.local_model.num_output_features == 2:
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                "Invalid number of output features: {}".format(
                    self.local_model.num_output_features
                )
            )

        if device.type == "cuda":
            print(
                f"Using device: {device}, GPU name: {torch.cuda.get_device_name(device.index)}"
            )
        else:
            print(f"Using device: {device}")

        def compute_loss(outputs, labels):
            if self.local_model.num_output_features == 1:
                return criterion(outputs, labels)
            elif self.local_model.num_output_features == 2:
                return criterion(outputs, labels.squeeze().long())
            else:
                raise ValueError(
                    "Invalid number of output features: {}".format(
                        self.local_model.num_output_features
                    )
                )

        def to_predictions(outputs):
            if self.local_model.num_output_features == 1:
                return (torch.sigmoid(outputs) > 0.5).float()
            elif self.local_model.num_output_features == 2:
                return outputs.argmax(dim=1)
            else:
                raise ValueError(
                    "Invalid number of output features: {}".format(
                        self.local_model.num_output_features
                    )
                )

            # Evaluate the self.local_model on the test data

        def evaluate(data_test_device):
            self.local_model.eval()  # Set the self.local_model to evaluation mode
            with torch.no_grad():  # Do not calculate gradients to save memory
                outputs_test = self.local_model(data_test_device)

                predictions_test = to_predictions(outputs_test)

                # Calculate metrics
                accuracy = accuracy_score(
                    data_test_device.y.cpu(), predictions_test.cpu()
                )
                precision = precision_score(
                    data_test_device.y.cpu(), predictions_test.cpu()
                )
                recall = recall_score(data_test_device.y.cpu(), predictions_test.cpu())
                f1 = f1_score(data_test_device.y.cpu(), predictions_test.cpu())
                return accuracy, precision, recall, f1

        # 训练循环
        accuracies = []
        best_accuracy = 0.0
        best_model_state_dict = None
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        for epoch in range(num_epochs):
            self.local_model.train()  # Set the model to training mode
            outputs = self.local_model(self.data_device)
            loss = compute_loss(outputs, self.data_device.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for param in self.local_model.parameters():
                    param.add_(torch.randn_like(param) * 0.1)  # 添加噪声

            print("Malicious behavior: Model quality has been intentionally degraded.")


            # Evaluate
            accuracy, precision, recall, f1 = evaluate(self.data_test_device)
            accuracies.append(accuracy)
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Loss: {loss.item()}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            self.accuracy = accuracy
            self.precision = precision
            self.recall = recall
            self.f1 = f1

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state_dict = copy.deepcopy(self.local_model.state_dict())
                best_precision = precision
                best_recall = recall
                best_f1 = f1

        # Find the maximum accuracy and its corresponding epoch
        max_accuracy = max(accuracies)
        max_epoch = accuracies.index(max_accuracy) + 1

        # Print the coordinates of the maximum point
        print(
            f"learning rate {learning_rate}, epoch {num_epochs} and dimension {num_output_features},Maximum accuracy of {100*max_accuracy:.2f}% at epoch {max_epoch}"
        )

        # plot_accuracy_vs_epoch(accuracies, num_epochs, learning_rate, self.local_model)
        self.local_model.load_state_dict(best_model_state_dict)
        # Save the best metrics to self
        self.accuracy = round(best_accuracy, 4)
        self.precision = round(best_precision, 4)
        self.recall = round(best_recall, 4)
        self.f1 = round(best_f1, 4)

    def upload_local_model(self, central_server_ip):
        # 序列化本地模型
        local_model_serialized = pickle.dumps(self.local_model)
        local_model_serialized_base64 = base64.b64encode(
            local_model_serialized
        ).decode()

        # 评估本地模型性能 ,来自上述训练的 accuracy, precision, recall, f1

        performance = self.accuracy, self.precision, self.recall, self.f1
        print(self.accuracy)

        # 发送本地模型及其性能到中心服务器
        response = requests.post(
            f"http://{central_server_ip}/upload_model",
            data={
                "drone_id": self.drone_id,
                "local_model": local_model_serialized_base64,
                "performance": self.accuracy,
            },
        )

        # print("Response status code:", response.status_code)
        # print("Response content:", response.text)

        if response.json()["status"] == "success":
            print(f"Drone {self.drone_id}: Model uploaded successfully.")
        else:
            print(f"Drone {self.drone_id}: Model upload failed.")

    def registerToMaster(self):
        print("连接到主节点……,本节点端口：" + str(self.port) + "\n")
        response = requests.post(
            f"http://{self.central_server_ip}/register",
            data={"drone_id": str(self.drone_id), "ip": "localhost:" + str(self.port)},
        )
        # print("Response status code:", response.status_code)
        # print("Response content:", response.text)
        # print("主节点连接建立结束……\n")

    def config(self, drone_id, local_data):
        self.drone_id = drone_id
        self.local_data = local_data

    def run(self):
        app = Flask(__name__)
        """
        run起来之后数据会丢失，一定要重新定义，获取self的变量， winodws的线程问题，暂时没有
        办法解决
        
        
        """
        torch.manual_seed(0)
        self.data_device = torch.load(f"data_object/node_train_{drone_id}.pt").to(
            device
        )
        self.data_test_device = torch.load(f"data_object/node_test_{drone_id}.pt").to(
            device
        )

        @app.route("/health_check", methods=["POST"])
        def health_check():
            # drone_id = request.form['drone_id']
            return jsonify({"status": "OK"})

        @app.route("/receive_model", methods=["POST"])
        def receiveModel():
            model_serialized_base64 = request.form["model"]
            model_serialized = base64.b64decode(model_serialized_base64)

            # Load the model's state_dict from the serialized byte stream
            buffer = io.BytesIO(model_serialized)
            state_dict = torch.load(buffer)

            # Create a new model and load the state_dict into it
            model = Net18(num_output_features).to(device)
            model.load_state_dict(state_dict)

            self.receive_global_model(model)
            print("LOGGER-INFO: global model received")

            # 先看看接受的模型准不准确
            def to_predictions(outputs):
                if self.local_model.num_output_features == 1:
                    return (torch.sigmoid(outputs) > 0.5).float()
                elif self.local_model.num_output_features == 2:
                    return outputs.argmax(dim=1)
                else:
                    raise ValueError(
                        "Invalid number of output features: {}".format(
                            self.local_model.num_output_features
                        )
                    )

            self.local_model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Do not calculate gradients to save memory
                outputs_test = self.local_model(self.data_test_device)

                predictions_test = to_predictions(outputs_test)

                # Calculate metrics
                accuracy = accuracy_score(
                    self.data_test_device.y.cpu(), predictions_test.cpu()
                )
                precision = precision_score(
                    self.data_test_device.y.cpu(), predictions_test.cpu()
                )
                recall = recall_score(
                    self.data_test_device.y.cpu(), predictions_test.cpu()
                )
                f1 = f1_score(self.data_test_device.y.cpu(), predictions_test.cpu())
                print(f"Accuracy of received model: {accuracy}")
                print(f"Precision of received model: {precision}")
                print(f"Recall of received model: {recall}")
                print(f"F1 Score of received model: {f1}")

            print("接收到全局模型，训练中")
            self.train_local_model()
            print("发送本地训练结果至主节点……")
            self.upload_local_model(self.central_server_ip)
            print("发送完毕……")

            return jsonify({"status": "OK"})

        # @app.route("/train", methods=["GET"])
        # def train():
        #     self.train_local_model()
        #     return jsonify({"status": "train finished"})

        @app.route("/uploadToMaster", methods=["POST"])
        def uploadToMaster(ip=self.central_server_ip):
            ip = request.json["ip"]
            self.upload_local_model(ip)
            return jsonify({"status": "upload to master succeed"})

        app.run(host="localhost", port=self.port)


if __name__ == "__main__":
    drone_id = int(sys.argv[2])

    drone_node_instance = MaliciousDroneNode(drone_id)
    drone_node_instance.port = sys.argv[1]

    drone_node_instance.drone_id = drone_id
    # 初次连接，接收全局模型，先训练一次
    p1 = Process(target=drone_node_instance.registerToMaster)
    p1.start()
    # drone_node_instance.registerToMaster()
    drone_node_instance.run()
