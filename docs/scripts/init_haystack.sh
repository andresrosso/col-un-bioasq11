#!/bin/bash
conda init bash
conda activate haystack
sleep 1
cd /opt/bioasq/haystack
nohup docker-compose up > /home/andresr/logs/haystack_docker.log &
echo ¨Docker compose started¨
sleep 10
