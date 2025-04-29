#!/bin/bash

apt update
apt install python3 python3-pip
su - storm -c "pip install gymnasium matplotlib requests prometheus-api-client"
