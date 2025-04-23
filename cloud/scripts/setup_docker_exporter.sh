#!/bin/bash

su - storm -c 'git clone https://github.com/Shayan-Ghani/Container-exporter.git /home/storm/Container-exporter'
su - storm -c 'docker compose -f /home/storm/Container-exporter/container-exporter.yml up -d'
