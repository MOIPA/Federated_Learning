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

class Aggregate:
    """聚合进程：
        1. 决定聚合模型时刻
        2. 负责和子节点通信，分发聚合模型
    """
    def __init__(self):
        # self.global_model = None
        # self.aggregated_global_model = None
        pass
    
    def check_and_aggregate_models(self, use_reputation=True):
        """聚合核心算法
            1. 提供全部模型聚合结果
            2. 提供声誉筛选模型聚合结果
            3. 绘图功能
        """
        aggregation_accuracies = []
        num_aggregations = 0 # 记录聚合次数

        kernel.setTorchSeed(0)
        print("LOGGER-INFO: check_and_aggregate_models() is called")
        start_time = time.time()
        # 模型准确度集合
        all_individual_accuracies = []
        aggregation_times = []
        # 检查队列中是否有足够的模型进行聚合·
        while True:
            # 定时守护线程，只用于聚合，每隔1s判断一次是否能够聚合
            time.sleep(1)
            modelNums = redisUtil.getListLength('modelList')
            # modelNums = self.local_models.qsize()
            if modelNums >= 2:
                redisUtil.normalSetDict('num_aggregations',num_aggregations)
                models_to_aggregate = []
                try:
                    # 使用声誉需要排序选择模型
                    if use_reputation:
                        # 获取所有模型和它们的声誉
                        all_modelsJson=redisUtil.batchReadFromList('modelList',2)[0]
                        # 解析下redis中获取的数据，从base64中反序列化过来
                        all_models = []
                        for item in all_modelsJson:
                            dictItem = eval(item)
                            dictItem = {key:kernel.serializeloadModelFromBase64(value) for key,value in dictItem.items()}
                            all_models.append(dictItem)
                        # all_models = [self.local_models.get() for _ in range(2)]  # 这里没有获取全部，只获取了两个
                        all_reputations = [
                            redisUtil.hgetNumber('reputation',drone_id)
                            #self.reputation[drone_id]
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
                        print(f"Current reputations: {self.getReputation()}")

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
                        all_modelsJson=redisUtil.batchReadFromList('modelList',2)[0]
                        # 解析下redis中获取的数据，从base64中反序列化过来
                        all_models = []
                        for item in all_modelsJson:
                            dictItem = eval(item)
                            dictItem = {key:kernel.serializeloadModelFromBase64(value) for key,value in dictItem.items()}
                            models_to_aggregate.append(dictItem)
                        # for _ in range(2):
                        #     model_dict = self.local_models.get()
                        #     models_to_aggregate.append(model_dict)
                finally:
                    pass
                # 评估模型准确度，加入集合
                individual_accuracies = []
                for model_dict in models_to_aggregate:
                    for drone_id, model in model_dict.items():
                        accuracy, precision, recall, f1 = kernel.fed_evaluate(model)
                        individual_accuracies.append(accuracy)
                all_individual_accuracies.append(individual_accuracies)
                
                try:
                    aggregated_global_model = kernel.aggregate_models(models_to_aggregate, use_reputation,self.getReputation())
                finally:
                    pass

                aggregation_times.append(time.time())

                accuracy, precision, recall, f1 = kernel.fed_evaluate(aggregated_global_model)

                print(f"Aggregated model accuracy after aggregation: {accuracy}")

                aggregation_accuracies.append(accuracy)
                num_aggregations += 1  # 增加模型聚合的次数
                print("Aggregation accuracies so far: ", aggregation_accuracies)

                drone_nodes = redisUtil.hgetAllConvertStringAndString('drone_nodes')
                for drone_id, ip in drone_nodes.items():
                    self.send_model_thread(ip,aggregated_global_model)

                # 当有10条记录就开始动态画图
                if num_aggregations == 20:
                    end_time = time.time()  # 记录结束时间
                    print(
                        f"Total time for aggregation: {end_time - start_time} seconds"
                    )  # 打印执行时间

                    # Find the time of the aggregation with the highest accuracy
                    max_accuracy_index = aggregation_accuracies.index(
                        max(aggregation_accuracies)
                    )
                    max_accuracy_time = aggregation_times[max_accuracy_index]
                    print(
                        f"Time of the aggregation with the highest accuracy: {max_accuracy_time - start_time} seconds"
                    )

                    plot_accuracy_vs_epoch(
                        aggregation_accuracies,
                        all_individual_accuracies,
                        num_aggregations,
                        learning_rate=0.02,
                    )
                    # 记录aggregation_accuracies
                    redisUtil.normalSetDict('aggregation_accuracies',aggregation_accuracies)

                    print("******************聚合完成，聚合进程终止******************")

                    sys.exit()  # Terminate the program

    def send_model_thread(self, ip,aggregatedGlobalModel):
        """发送模型线程，因为发送模型任务较为消耗资源，具体执行任务交给子线程执行，主线程专注于提供服务"""
        threading.Thread(target=self.send_model, args=(ip,aggregatedGlobalModel)).start()

    def getReputation(self):
        res = redisUtil.hgetAll('reputation')
        return {key.decode('utf-8'):float(value) for key,value in res.items()}

    def send_model(self, ip,aggregated_global_model):
        """发送模型到目标节点服务，若无全局模型需先训练"""
        global_model_serialized = kernel.saveModelAndGetValue(aggregated_global_model.state_dict())

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
        
    def run(self):
        threading.Thread(target=self.check_and_aggregate_models).start()

agg = Aggregate()
agg.run()