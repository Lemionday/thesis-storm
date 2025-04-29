#!/bin/bash

apt install -y python3-pip

su - storm -c "python3 -m pip install --user ansible-core"

su - storm -c 'cp /home/storm/thesis-storm/storm-exporter/.env.example /home/storm/thesis-storm/storm-exporter/.env'
# echo "storm.local.hostname: nimbus.thesis-storm_default" >> /home/storm/thesis-storm/config/storm.yml
su - storm -c 'docker compose -f /home/storm/thesis-storm/docker-compose.cloud.yml up -d'
mkdir -p /home/storm/thesis-storm/stormsmarthome/target
chown storm:storm -R /home/storm/thesis-storm/stormsmarthome/target
