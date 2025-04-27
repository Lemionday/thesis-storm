# ⛈️ Multi-Level Cloud Resource Autoscaling with Apache Storm

This repository contains the thesis implementation for an intelligent
autoscaling system tailored for **Apache Storm** on the cloud. The system
comprises three core components:

1. **Storm Exporter** – a Prometheus-compatible metrics exporter
2. **Forecast** – a predictive module that analyzes resource trends
3. **Autoscaler** – a controller that dynamically adjusts computing resources

The architecture is built to support both **local testing** and **GCP
deployment** environments.

---

## 🌀 Repository Structure

```bash
.
├── storm-exporter/     # Custom Prometheus exporter for Apache Storm metrics
├── forecast/           # Predictive analytics module (e.g., using stats or ML)
├── autoscaler/         # Dynamic scaling logic based on forecast and metrics
├── docker-compose.yml  # Local orchestration
└── terraform/          # GCP provisioning (VMs, Docker, firewall, etc.)

```

### GCP

```bash
# master
docker compose -f /home/storm/thesis-storm/docker-compose.cloud.supervisor.yml up -d

cp storm-exporter/.env.example storm-exporter/.env

docker compose -f /home/storm/thesis-storm/docker-compose.cloud.supervisor.yml up -d

cd cloud
just copy-ssh-key

eval $(ssh-agent)
ssh-add ~/.ssh/storm

ssh -i .ssh/storm 10.148.0.22

ansible supervisor -i inventory -m ping

ansible all -i inventory.ini -m ansible.builtin.copy -a "src=./docker-compose.cloud.supervisor.yml dest=~/thesis-storm/docker-compose.cloud.supervisor.yml"
ansible all -i inventory.ini -m community.docker.docker_compose_v2 -a "project_src=~/thesis-storm/docker-compose.cloud.supervisor.yml state=restarted"

```
