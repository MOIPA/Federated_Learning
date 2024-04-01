import logging
import os

# 检查日志目录
folder_name = "log"
path = "./" + folder_name
os.makedirs(path, exist_ok=True)

class InfoLogger():
    """抽离日志模块，便于以后自定义输出功能和输出方向"""
    def __init__(self,id) -> None:
        self.id = id
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # 创建文件处理器并设置日志级别为DEBUG
        file_handler = logging.FileHandler("log/id_"+str(self.id)+"_service.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        # 将文件处理器添加到日志对象中
        self.logger.addHandler(file_handler)
        
        # 打印不同级别的日志信息
        self.logger.debug("日志模块初始化完成")

    def log(self,msg):
        self.logger.debug(msg)