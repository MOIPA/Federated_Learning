from flask import Flask, request, jsonify
import pickle
import base64
import requests
import time
import os
from queue import Queue
import threading
import json
from tools import plot_accuracy_vs_epoch
import sys
from Algorithm import KernelAlgo

deviceConfig = 'cpu' 
kernel = KernelAlgo(map_location=deviceConfig)

class CentralServer:
    """中心服务器：
        1. 用于维护节点模型声望值，决定聚合模型时刻
        2. 负责和子节点通信，分发聚合模型
    """
    def __init__(self):
        self.global_model = None
        self.aggregated_global_model = None
        self.reputation = {}
        self.local_models = {}
        self.local_models = Queue()  # 使用队列来存储上传的模型
        self.lock = threading.Lock()  # 创建锁
        self.aggregation_method = "async weighted  aggregation"
        self.new_model_event = threading.Event()
        self.drone_nodes = {}
        self.aggregation_accuracies = []
        self.num_aggregations = 0  # 记录聚合次数
        self.data_age = {}
        self.low_performance_counts = {}

        threading.Thread(target=self.check_and_aggregate_models).start()

    def check_and_aggregate_models(self, use_reputation=True):
        """聚合核心算法
            1. 提供全部模型聚合结果
            2. 提供声誉筛选模型聚合结果
            3. 绘图功能
        """
        kernel.setTorchSeed(0)
        print("LOGGER-INFO: check_and_aggregate_models() is called")
        start_time = time.time()
        # 模型准确度集合
        all_individual_accuracies = []
        aggregation_times = []
        # 检查队列中是否有足够的模型进行聚合·
        while True:
            # 沉睡等待触发聚合信号
            self.new_model_event.wait()
            if self.local_models.qsize() >= 2:
                models_to_aggregate = []
                # 获取锁防止多线程同时访问队列导致脏读
                self.lock.acquire()
                try:
                    # 使用声誉需要排序选择模型
                    if use_reputation:
                        # 获取所有模型和它们的声誉
                        all_models = [self.local_models.get() for _ in range(2)]
                        all_reputations = [
                            self.reputation[drone_id]
                            for model_dict in all_models
                            for drone_id in model_dict
                        ]
                        # 根据声誉排序模型
                        sorted_indices = sorted(
                            range(len(all_reputations)),
                            key=lambda i: all_reputations[i],
                            reverse=True,
                        )
                        # 选择声誉最高的3个模型
                        models_to_aggregate = [
                            all_models[i] for i in sorted_indices[:2]
                        ]

                        # 打印每轮的所有节点声誉
                        print("                   ")
                        print(f"Current reputations: {self.reputation}")

                        # 打印参与聚合的节点
                        aggregated_node_ids = [
                            list(model_dict.keys())[0]
                            for model_dict in models_to_aggregate
                        ]
                        print(
                            f"Nodes participating in aggregation: {aggregated_node_ids}"
                        )

                    else:
                        # 如果不使用声誉，那么就选择所有的模型
                        for _ in range(2):
                            model_dict = self.local_models.get()
                            models_to_aggregate.append(model_dict)
                finally:
                    self.lock.release()
                # 评估模型准确度，加入集合
                individual_accuracies = []
                for model_dict in models_to_aggregate:
                    for drone_id, model in model_dict.items():
                        accuracy, precision, recall, f1 = kernel.fed_evaluate(model)
                        individual_accuracies.append(accuracy)
                all_individual_accuracies.append(individual_accuracies)
                
                self.lock.acquire()  # Acquire lock before aggregation
                try:
                    self.aggregated_global_model = kernel.aggregate_models(models_to_aggregate, use_reputation,self.reputation)
                finally:
                    self.lock.release()

                aggregation_times.append(time.time())

                accuracy, precision, recall, f1 = kernel.fed_evaluate(self.aggregated_global_model)

                print(f"Aggregated model accuracy after aggregation: {accuracy}")

                self.aggregation_accuracies.append(accuracy)
                self.num_aggregations += 1  # 增加模型聚合的次数
                print("Aggregation accuracies so far: ", self.aggregation_accuracies)

                for drone_id, ip in self.drone_nodes.items():
                    self.send_model_thread(ip, "aggregated_global_model")

            # 当有10条记录就开始动态画图
            if self.num_aggregations == 20:
                end_time = time.time()  # 记录结束时间
                print(
                    f"Total time for aggregation: {end_time - start_time} seconds"
                )  # 打印执行时间

                # Find the time of the aggregation with the highest accuracy
                max_accuracy_index = self.aggregation_accuracies.index(
                    max(self.aggregation_accuracies)
                )
                max_accuracy_time = aggregation_times[max_accuracy_index]
                print(
                    f"Time of the aggregation with the highest accuracy: {max_accuracy_time - start_time} seconds"
                )

                plot_accuracy_vs_epoch(
                    self.aggregation_accuracies,
                    all_individual_accuracies,
                    self.num_aggregations,
                    learning_rate=0.02,
                )

                print("******************聚合完成，聚合进程终止******************")

                sys.exit()  # Terminate the program
            self.new_model_event.clear()

    def update_reputation(self, drone_id, new_reputation):
        self.reputation[drone_id] = new_reputation

    def send_model(self, ip, model_type="global_model"):
        """发送模型到目标节点服务，若无全局模型需先训练"""
        # 根据 model_type 加载不同的模型
        if model_type == "aggregated_global_model":
            model_path = "aggregated_global_model.pt"

            global_model_serialized = kernel.saveModelAndGetValue(self.aggregated_global_model.state_dict())

            # Encode the bytes as a base64 string
            global_model_serialized_base64 = base64.b64encode(
                global_model_serialized
            ).decode()
            # 发送模型到子节点服务器
            # print("发送全局模型-发送！--->" + ip)
            s = requests.session()
            s.keep_alive = False
            response = requests.post(
                f"http://{ip}/receive_model",
                data={"model": global_model_serialized_base64},
            )
            # print("发送全局模型-成功！--->" + ip)
            return json.dumps({"status": "success"})
        else:  # 默认加载 global_model
            model_path = "global_model.pt"
            # 检查模型是否已经存在
            if os.path.isfile(model_path):
                # 加载模型
                self.global_model = kernel.loadModel(model_path,deviceConfig)
                print(f"本地存在 {model_type}，发送中…… ")
            else:
                # 训练新模型
                print(f"本地不存在 {model_type}，训练中……")
                kernel.initialize_global_model()

            global_model_serialized = kernel.saveModelAndGetValue(self.global_model.state_dict())
            # Encode the bytes as a base64 string
            global_model_serialized_base64 = base64.b64encode(
                global_model_serialized
            ).decode()
            # 发送模型到子节点服务器
            # print("发送全局模型-发送！--->" + ip)
            requests.post(
                f"http://{ip}/receive_model",
                data={"model": global_model_serialized_base64},
            )
            print("发送全局模型-成功！--->" + ip)
            return json.dumps({"status": "success"})

    def send_model_thread(self, ip, model_type="aggregated_global_model"):
        """发送模型线程，因为发送模型任务较为消耗资源，具体执行任务交给子线程执行，主线程专注于提供服务"""
        threading.Thread(target=self.send_model, args=(ip, model_type)).start()

    # "0.0.0.0"表示应用程序在所有可用网络接口上运行
    def run(self, port=5000):
        app = Flask(__name__)
        print(kernel.criterion)

        @app.route("/health_check", methods=["GET"])
        def health_check():
            return jsonify({"status": "OK"})

        @app.route("/register", methods=["POST"])
        def register():
            drone_id = request.form["drone_id"]
            ip = request.form["ip"]
            print("接收到新节点,id:" + drone_id + ",ip:" + ip)
            # 将新的无人机节点添加到字典中
            self.drone_nodes[drone_id] = ip
            print("发送全局模型-执行中--->" + ip)
            self.send_model_thread(ip, "global_model")
            return jsonify({"status": "success"})

        @app.route("/upload_model", methods=["POST"])
        def upload_model():
            drone_id = request.form["drone_id"]
            local_model_serialized = request.form["local_model"]
            local_model = pickle.loads(base64.b64decode(local_model_serialized))
            local_model_serialized = pickle.dumps(local_model)

            performance = float(request.form["performance"])

            # 更新数据时效性标签
            if drone_id in self.data_age:
                self.data_age[drone_id] += 1
            else:
                self.data_age[drone_id] = 1

            reputation = kernel.compute_reputation(
                drone_id,
                performance,
                self.data_age[drone_id],
                self.low_performance_counts,
                performance_threshold=0.7,
            )

            self.update_reputation(drone_id, reputation)

            self.local_models.put({drone_id: local_model})

            self.new_model_event.set()

            return jsonify({"status": "success"})

        app.run(host="localhost", port=port)


central_server_instance = CentralServer()
central_server_instance.run()
