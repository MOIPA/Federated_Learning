import redis
from Logger import InfoLogger
import json
from Algorithm import KernelAlgo

class MyRedis():
    """
        redis操作模块，封装写入，读取，删除，弹出等操作以及各自的批量操作
    """

    """
        以下为list操作
    """
    def __init__(self) -> None:
        # 初始化连接件
        self.logger = InfoLogger('redis')
        self.r = redis.StrictRedis(host='localhost', port=6379)
        self.logger.log("连接redis初始化完成")
    
    # list中增加
    def pushToModelList(self,listName,element):
        self.r.lpush(listName,element)
    
    def insertToModelList(self,listName,element,newElement):
        self.r.linsert(listName, 'BEFORE', element, newElement)    # 在目标值之前插入新值

    # 获取list的长度
    def getListLength(self,listName):
        return self.r.llen(listName)
    
    # 获取list中的某个
    def getElementFromList(self,listName,index):
        return self.r.lindex(listName,index)
    
    # 读取并删除前n个数据 range从1开始算
    def batchReadFromList(self,listName,range):
        p = self.r.pipeline()
        p.lrange(listName, 0, range - 1)
        p.ltrim(listName, range, -1)
        data = p.execute()
        return data
    
    """
        以下为字典操作和普通kv操作
    """
    def normalSet(self,k,v):
        self.r.set(k,v)
    
    """存储dict版本"""
    def normalSetDict(self,key,dict):
        json_data = json.dumps(dict)
        self.r.set(key,json_data)

    """读取dict版本"""
    def normalGetDict(self,key):
        json_data = self.r.get(key)
        return json.loads(json_data)

    def normalGet(self,k):
        return self.r.get(k)
    
    def hset(self,key,filed,value):
        self.logger.log("redis已执行哈希表更新")
        self.r.hset(key,filed,value)
    
    def hget(self,key,filed):
        return self.r.hget(key,filed)
    
    def hgetNumber(self,key,filed):
        return float(self.r.hget(key,filed))
    
    def hgetAll(self,key):
        return self.r.hgetall(key)
    
    def hlen(self,key):
        return self.r.hlen(key)
    
    """hgetAll的转换版本，key为string类型，value为float类型"""
    def hgetAllConvertStringAndFloat(self,key):
        res = self.r.hgetall(key)
        return {key.decode('utf-8'):float(value) for key,value in res.items()}
    
    """hgetAll的转换版本，key为string类型，value为int类型"""
    def hgetAllConvertStringAndInt(self,key):
        res = self.r.hgetall(key)
        return {key.decode('utf-8'):int(value) for key,value in res.items()}

    """hgetAll的转换版本，key为string类型，value为string类型"""
    def hgetAllConvertStringAndString(self,key):
        res = self.r.hgetall(key)
        return {key.decode('utf-8'):value.decode('utf-8') for key,value in res.items()}
    
    """清空redis"""
    def resetRedis(self):
        self.r.flushall()


if __name__ == "__main__":
    redisUtil = MyRedis()
    redisUtil.pushToModelList('model','1')
    redisUtil.pushToModelList('model','2')
    redisUtil.pushToModelList('model','3')
    redisUtil.pushToModelList('model','4')
    redisUtil.pushToModelList('model','5')
    len = redisUtil.getListLength('model')
    res = redisUtil.batchReadFromList('model',3)
    