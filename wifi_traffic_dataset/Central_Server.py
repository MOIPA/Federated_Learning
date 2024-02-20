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
from RedisUtils import MyRedis

deviceConfig = 'cpu' 
kernel = KernelAlgo(map_location=deviceConfig)
redisUtil = MyRedis()

class CentralServer:
    """中心服务器：
        1. 用于维护节点模型声望值，决定聚合模型时刻
        2. 负责和子节点通信，分发聚合模型
    """
    def __init__(self):
        # self.global_model = None
        # self.aggregated_global_model = None
        # self.reputation = {}
        # self.local_models = {}
        # self.local_models = Queue()  # 使用队列来存储上传的模型
        # self.lock = threading.Lock()  # 创建锁
        # self.aggregation_method = "async weighted  aggregation"
        # self.new_model_event = threading.Event()
        #self.low_performance_counts = {}
        # threading.Thread(target=self.check_and_aggregate_models).start()
        pass

    def getReputation(self):
        res = redisUtil.hgetAll('reputation')
        return {key.decode('utf-8'):float(value) for key,value in res.items()}

    def update_reputation(self, drone_id, new_reputation):
        redisUtil.hset('reputation',drone_id,new_reputation)
        # self.reputation[drone_id] = new_reputation

    def send_global_model(self, ip):
        """发送模型到目标节点服务，若无全局模型需先训练"""
        # 根据 model_type 加载不同的模型
        # 默认加载 global_model
        model_path = "global_model.pt"
        # 检查模型是否已经存在
        if os.path.isfile(model_path):
            # 加载模型
            global_model = kernel.loadModel(model_path,deviceConfig)
            print(f"本地存在global_model，发送中…… ")
        else:
            # 训练新模型
            print(f"本地不存在global_model，训练中……")
            kernel.initialize_global_model()

        global_model_serialized = kernel.saveModelAndGetValue(global_model.state_dict())
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

    def send_global_model_thread(self, ip):
        """发送模型线程，因为发送模型任务较为消耗资源，具体执行任务交给子线程执行，主线程专注于提供服务"""
        threading.Thread(target=self.send_global_model, args=(ip,)).start()

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
            redisUtil.hset('drone_nodes',drone_id,ip)
            #self.drone_nodes[drone_id] = ip
            print("发送全局模型-执行中--->" + ip)
            self.send_global_model_thread(ip)
            return jsonify({"status": "success"})

        @app.route("/upload_model", methods=["POST"])
        def upload_model():
            drone_id = request.form["drone_id"]
            local_model_serialized = request.form["local_model"]
            local_model = kernel.serializeloadModelFromBase64(local_model_serialized)
            # local_model = pickle.loads(base64.b64decode(local_model_serialized)) 
            performance = float(request.form["performance"])

            # 更新数据时效性标签
            data_age = redisUtil.hgetAllConvertStringAndInt('data_age')
            if drone_id in data_age:
                data_age[drone_id] += 1
                redisUtil.hset('data_age',drone_id,data_age[drone_id])
            else:
                data_age[drone_id] = 1
                redisUtil.hset('data_age',drone_id,data_age[drone_id])
            low_performance_counts = {}
            reputation = kernel.compute_reputation(
                drone_id,
                performance,
                data_age[drone_id],
                low_performance_counts,
                performance_threshold=0.7,
            )

            self.update_reputation(drone_id, reputation)

            redisUtil.pushToModelList('modelList',{drone_id:local_model_serialized})
            redisUtil.hset('modelSet',drone_id,local_model_serialized)
            # self.local_models.put({drone_id: local_model})

            # self.new_model_event.set()

            return jsonify({"status": "success"})

        @app.route("/reset", methods=["GET"])
        def resetRedis():
            redisUtil.resetRedis()

        app.run(host="localhost", port=port)

# 每次启动前先清理redis
redisUtil.resetRedis()
central_server_instance = CentralServer()
central_server_instance.run()
