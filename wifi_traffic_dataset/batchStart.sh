#!/bin/bash
## This is the script to startup a batch of DroneNode instances
## format of instances can be DroneNode <node id> <port>
## Writen by TR 2023-01-19

date
echo "Starting up instances"
num=0
idStart=1
portStart=50001
read -p "请输入需要启动的节点数量:" num
read -p "请输入需要启动的节点id编号起始位置（默认为1）:" idStart
read -p "请输入需要启动的节点端口编号起始位置（默认为50001）:" portStart

if [ -z "$num" ]; then
	num=10
fi
if [ -z "$idStart" ]; then
	idStart=1
fi
if [ -z "$portStart" ]; then
	portStart=50001
fi

echo "启动节点数量：$num" 
echo "id起始地址：$idStart" 
echo "port起始地址：$portStart" 

for i in `seq 0 $(($num-1))`; do
	port=$((i+portStart))
	id=$((i+idStart))
	echo "启动实例中,id:$id , port:$port"
	logPath="./log/flask_id_$id.log"
	echo $logPath
	python ./Drone_Node.py $port $id > $logPath &
done

python ./Central_Server.py &
