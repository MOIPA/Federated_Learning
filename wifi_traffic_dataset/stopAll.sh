#!/bin/bash

process_info=$(ps -ef | grep "Drone_Node") # 将"进程名"替换为需要查找的进程名称或关键字

pid=$(echo "$process_info" | awk '{print $2}')

if [ ! -z "$pid" ]; then
    echo "Killing process with PID: $pid"
    kill -9 $pid
else
    echo "No matching processes found."
fi

pkill -fi Central_Server.py