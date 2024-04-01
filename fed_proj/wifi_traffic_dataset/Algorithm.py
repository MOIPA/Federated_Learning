import numpy as np
from Model import Net18
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import copy
import base64
from aggregation_solution import weighted_average_aggregation, average_aggregation
from tools import sigmoid, exponential_decay
import pickle


class KernelAlgo():
    """
    算法模块，封装了联邦学习的核心算法  
    ****Note****
    注意这里的map_location不传的话是默认gpu，如果需要使用cpu传递cpu即可
    """
    def __init__(self,map_location='gpu'
                 ,serverTrainPath="data_object/server_train.pt"
                 ,serverTestPath="data_object/server_test.pt"
                 ,learning_rate=0.01,num_output_features=2,seed=0,performance_threshold=0.7) -> None:
        """
            模块初始化，获取以下成员变量：
                data_device         训练数据
                data_test_device    测试数据
                criterion           损失函数
                device              计算设备
        """
        self.learning_rate = learning_rate
        self.num_output_features = num_output_features
        # 初始化torch设置，配置使用的设备类型（cpu/gpu）,配置损失函数
        self.initTorch(seed)
        # 初始化训练和采集数据，封装私有变量
        self.initData(map_location,serverTrainPath,serverTestPath)

    def saveModelAndGetValue(self,state_dict):
        """保存模型参数字典，并且返回保存结果
            @state_dict     目标模型的参数字典
        """
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()
    
    def serializeModel(self,model):
        """序列化模型，model为Net18 pytorch类型，通过序列化关键stat_dict模型参数，做到序列化和重载
        """
        serialized = self.saveModelAndGetValue(model.state_dict())
            # Encode the bytes as a base64 string
        base64 = base64.b64encode(
            serialized
        ).decode()
        return base64
    
    def serializeloadModelFromBase64(self,base64Info):
        """反序列化模型，装载模型到内存"""
        local_model = pickle.loads(base64.b64decode(base64Info))
        return local_model

    def loadModel(self,model_path,map_location="gpu"):
        """加载参数，返回新模型实例
            @model_path     模型参数路径
            @map_location   执行设备
        """
        model = self.getNet18Model()
        if map_location!='cpu':
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path,map_location='cpu'))
        return model

    def initTorch(self,seed):
        """初始化torch的配置"""
        torch.manual_seed(seed)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initData(self,location,serverTrainPath,serverTestPath):
        """初始化数据集配置"""
        if location!='cpu':
            server_train = torch.load(serverTrainPath)
            server_test = torch.load(serverTestPath)   
            self.data_device = server_train.to(self.device)
            self.data_test_device = server_test.to(self.device)
        else:
            server_train = torch.load(serverTrainPath,map_location = location)
            server_test = torch.load(serverTestPath,map_location = location)        
            self.data_device = server_train.to(self.device)
            self.data_test_device = server_test.to(self.device)
    
    def to_predictions(self, outputs):
        """激活函数：argmax函数，最大值自变量集，输出最大概率分类"""
        return outputs.argmax(dim=1)

    def compute_loss(self, outputs, labels):
        """通过损失函数计算实际值和计算值的偏差"""
        return self.criterion(outputs, labels.squeeze().long())

    def fed_evaluate(self,model):
            """联邦学习评估"""
            model.eval()
            with torch.no_grad():  # Do not calculate gradients to save memory
                outputs_test = model(self.data_test_device)
                predictions_test = self.to_predictions(outputs_test)
                # Calculate metrics
                # Calculate metrics
                accuracy = round(
                    accuracy_score(self.data_test_device.y.cpu(), predictions_test.cpu()), 4
                )
                precision = round(
                    precision_score(self.data_test_device.y.cpu(), predictions_test.cpu()), 4
                )
                recall = round(
                    recall_score(self.data_test_device.y.cpu(), predictions_test.cpu()), 4
                )
                f1 = round(f1_score(self.data_test_device.y.cpu(), predictions_test.cpu()), 4)
                return accuracy, precision, recall, f1

    def compute_reputation(self, drone_id, performance, data_age,low_performance_counts, performance_threshold=0.7):
            """计算声望值，并且更新对应节点的声望值
                若本次模型性能小于阈值，对应的低声望值记录数量+1
                若本次模型性能大于阈值，计算惩罚因子重新计算声望，并更新声望值记录-1

                low_performance_counts  小于阈值的低声望值字典 droneId:<声望>
            """
            performance_contribution = sigmoid(performance)
            data_age_contribution = exponential_decay(data_age)
            reputation = performance_contribution * 0.9 + data_age_contribution * 0.1

            # 检查性能是否低于阈值，并更新连续低性能计数
            performance_threshold = 0.66
            # 检查并更新连续低性能计数
            if performance < performance_threshold:  # 你可以选择合适的阈值
                if drone_id in low_performance_counts:
                    low_performance_counts[drone_id] += 1
                else:
                    low_performance_counts[drone_id] = 1
                print("low performance node,", drone_id)
                reputation = reputation * 0.1
            else:
                penalty_factor = 1 / (
                    1 + np.exp(low_performance_counts.get(drone_id, 0))
                )

                if (
                    drone_id in low_performance_counts
                    and low_performance_counts[drone_id] > 0
                ):
                    low_performance_counts[drone_id] -= 1
                print("use penalty factor")
                reputation = reputation * penalty_factor
            return reputation
    
    def getNet18Model(self):
        """获取net18模型实例"""
        return Net18(self.num_output_features).to(self.device)

    def getOptimizer(self,model):
        """获取优化器实例"""
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate)
    
    def getDeviceName(self):
        """获取设备名称"""
        return torch.cuda.get_device_name(self.device.index)
    
    def printDevice(self):
        """打印设备信息"""
        if self.device.type == "cuda":
            print(
                f"Using device: {self.device}, GPU name: {self.getDeviceName()}"
            )
        else:
            print(f"Using device: {self.device}")

    def compute_loss_for_initialize_global_model(self, outputs, labels,model):
        """通过损失函数计算实际值和计算值的偏差"""
        if model.num_output_features == 1:
            return self.criterion(outputs, labels)
        elif model.num_output_features == 2:
            return self.criterion(outputs, labels.squeeze().long())
        else:
            raise ValueError(
                "Invalid number of output features: {}".format(
                    model.num_output_features
                )
            )
    
    def evaluate_for_initialize_global_model(self, outputs, labels,model):
        """评估"""
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Do not calculate gradients to save memory
            outputs_test = model(self.data_test_device)

            predictions_test = self.to_predictions_for_initialize_global_model(outputs_test,model)

            # Calculate metrics
            accuracy = accuracy_score(
                self.data_test_device.y.cpu(), predictions_test.cpu()
            )
            precision = precision_score(
                self.data_test_device.y.cpu(), predictions_test.cpu()
            )
            recall = recall_score(self.data_test_device.y.cpu(), predictions_test.cpu())
            f1 = f1_score(self.data_test_device.y.cpu(), predictions_test.cpu())
            return accuracy, precision, recall, f1
        
    def to_predictions_for_initialize_global_model(self, outputs,model):
        """激活函数：sigMod函数"""
        if model.num_output_features == 1:
            return (torch.sigmoid(outputs) > 0.5).float()
        elif model.num_output_features == 2:
            return outputs.argmax(dim=1)
        else:
            raise ValueError(
                "Invalid number of output features: {}".format(
                    model.num_output_features
                )
            )
        
    def initialize_global_model(self):
        """初始化：训练并返回全局模型"""
        num_epochs = 15
        model = self.getNet18Model()
        optimizer = self.getOptimizer(model)
        self.printDevice()
        # 训练模型并记录每个epoch的准确率
        accuracies = []
        best_accuracy = 0.0
        best_model_state_dict = None
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            outputs = model(self.data_device)
            loss = self.compute_loss_for_initialize_global_model(outputs, self.data_device.y,model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Evaluate
            accuracy, precision, recall, f1 = self.evaluate_for_initialize_global_model(self.data_test_device,model)
            accuracies.append(accuracy)
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"Loss: {loss.item()}")
            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state_dict = copy.deepcopy(model.state_dict())
        # After training, load the best model weights
        model.load_state_dict(best_model_state_dict)
        # Save the best model to a file
        torch.save(model.state_dict(), "global_model.pt")
        # Find the maximum accuracy and its corresponding epoch
        max_accuracy = max(accuracies)
        max_epoch = accuracies.index(max_accuracy) + 1

        # Print the coordinates of the maximum point
        print(
            f"learning rate {self.learning_rate}, epoch {num_epochs} and dimension {model.num_output_features},Maximum accuracy of {100*max_accuracy:.2f}% at epoch {max_epoch}"
        )
        return model
    
    def aggregate_models(self, models_to_aggregate, use_reputation,reputation):
        """聚合模型
            @models_to_aggregate    待聚合模型的集合
            @use_reputation         使用声誉开关
            @reputation             模型声誉字典
        """
        if use_reputation:
            aggregated_model = weighted_average_aggregation(
                models_to_aggregate, reputation
            )
        else:
            aggregated_model = average_aggregation(models_to_aggregate)

        aggregated_global_model = self.getNet18Model()  # 创建新的模型实例
        aggregated_global_model.load_state_dict(aggregated_model)  # 加载聚合后的权重

        # 保存全局模型到文件， 以pt形式保存
        torch.save(
            aggregated_global_model.state_dict(), "aggregated_global_model.pt"
        )
        return aggregated_global_model

    def setTorchSeed(self,seed=0):
        """设置torch随机种子"""
        torch.manual_seed(seed)

