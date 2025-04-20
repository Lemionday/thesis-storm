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
cp storm-exporter/.env.example storm-exporter/.env

docker compose -f /home/storm/thesis-storm/docker-compose.cloud.supervisor.yml up -d


ssh -i .ssh/storm 10.148.0.24 # Add remote host to known hosts
```
