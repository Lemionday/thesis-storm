#!/bin/bash

echo "storm.local.hostname: $(hostname)" >> /home/storm/thesis-storm/config/storm.cloud.yml
su - storm -c 'docker compose -f /home/storm/thesis-storm/docker-compose.cloud.supervisor.yml up -d'
