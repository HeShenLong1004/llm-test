#!/bin/bash

# 查找 gunicorn 进程
PID=$(ps aux | grep '/home/ecs-user/miniconda3/bin/python /home/ecs-user/miniconda3/bin/gunicorn -w 4 --bind 0.0.0.0:10000 llm_build_serving:app' | grep -v grep | awk '{print $2}')

# 如果找到了 PID，就结束进程
if [ ! -z "$PID" ]; then
    echo "Stopping gunicorn process with PID $PID"
    sudo kill -9 $PID
else
    echo "No gunicorn process found."
fi